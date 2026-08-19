#!/usr/bin/env python3
from __future__ import annotations

import json, math, queue, threading, tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Dict, List, Optional
from PIL import Image, ImageOps, ImageTk
from odb_cam_renderer import ODBError, ODBRenderer, contours_bounds, extract_input, parse_kv_blocks, parse_profile_contours

PREVIEW_DPI_DEFAULT=600; OUTPUT_DPI_DEFAULT=1200; MIN_DPI=72; MAX_DPI=10000
MIN_UM=0.1; MAX_UM=1000.0; ZOOM_MIN=.05; ZOOM_MAX=16.; ZOOM_STEP=1.25
MODE_DPI="DPI"; MODE_AOI="AOI 해상도"; CUSTOM="사용자 지정"
PROFILE_FILE=Path.home()/".odb_cam_renderer"/"aoi_profiles.json"

@dataclass(frozen=True)
class LayerInfo:
    name:str; layer_type:str="?"; context:str="?"; side:str="?"; polarity:str="?"
    @property
    def label(self):
        return f"{self.name}  [{self.layer_type}{' / '+self.side if self.side not in {'','?'} else ''}]"

@dataclass
class JobInfo:
    name:str; steps:List[str]; layers:List[LayerInfo]


def inspect_job(job:Path)->JobInfo:
    steps_dir=job/"steps"
    if not steps_dir.is_dir(): raise ODBError("ODB++ steps directory is missing")
    steps=sorted(p.name for p in steps_dir.iterdir() if p.is_dir()); layers=[]
    matrix=job/"matrix"/"matrix"
    if matrix.exists():
        for b in parse_kv_blocks(matrix,"LAYER"):
            if b.get("NAME"):
                layers.append(LayerInfo(b["NAME"],b.get("TYPE","?"),b.get("CONTEXT","?"),b.get("SIDE","?"),b.get("POLARITY","?")))
    return JobInfo(job.name,steps,layers)


def profile_size_mm(job:Path,step:str):
    b=contours_bounds(parse_profile_contours(job/"steps"/step.lower()/"profile"))
    return (b[2]-b[0])*25.4,(b[3]-b[1])*25.4


def load_profiles()->Dict[str,dict]:
    try: return json.loads(PROFILE_FILE.read_text(encoding="utf-8")) if PROFILE_FILE.exists() else {}
    except Exception: return {}


def save_profiles(data:Dict[str,dict]):
    PROFILE_FILE.parent.mkdir(parents=True,exist_ok=True)
    PROFILE_FILE.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")


class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("ODB++ CAM Image Renderer"); self.geometry("1380x850"); self.minsize(1120,700)
        self.job=None; self.tmp=None; self.info=None; self.source=None; self.preview=None; self.photo=None
        self.zoom=1.; self.offx=self.offy=0.; self.pan=None; self.preview_dpi=PREVIEW_DPI_DEFAULT; self.profiles=load_profiles(); self.token=0; self.q=queue.Queue()
        self.step=tk.StringVar(); self.layer=tk.StringVar(); self.mode=tk.StringVar(value=MODE_DPI); self.dpi=tk.IntVar(value=OUTPUT_DPI_DEFAULT)
        self.profile=tk.StringVar(value=CUSTOM); self.umx=tk.DoubleVar(value=10.); self.umy=tk.DoubleVar(value=10.); self.lock=tk.BooleanVar(value=True)
        self.zoom_text=tk.StringVar(value="100%"); self.status=tk.StringVar(value="ODB++ TGZ 파일을 열어주세요.")
        self._menu(); self._ui(); self._mode_changed(); self.protocol("WM_DELETE_WINDOW",self.close); self.after(100,self.poll)

    def _menu(self):
        m=tk.Menu(self); f=tk.Menu(m,tearoff=False); f.add_command(label="ODB++ 파일 열기...",accelerator="Ctrl+O",command=self.open_file); f.add_separator(); f.add_command(label="종료",command=self.close); m.add_cascade(label="파일",menu=f)
        s=tk.Menu(m,tearoff=False); s.add_command(label="미리보기 설정...",command=self.preview_settings); m.add_cascade(label="설정",menu=s); self.config(menu=m); self.bind_all("<Control-o>",lambda e:self.open_file())

    def _ui(self):
        root=ttk.Frame(self,padding=12); root.pack(fill="both",expand=True); top=ttk.Frame(root); top.pack(fill="x",pady=(0,10)); ttk.Button(top,text="파일 업로드",command=self.open_file).pack(side="left"); self.file_label=ttk.Label(top,text="선택된 파일 없음"); self.file_label.pack(side="left",padx=12)
        panes=ttk.Panedwindow(root,orient="horizontal"); panes.pack(fill="both",expand=True); left=ttk.Frame(panes,padding=(4,8)); right=ttk.Frame(panes,padding=(8,8)); panes.add(left,weight=0); panes.add(right,weight=1)
        ttk.Label(left,text="렌더링 설정",font=("TkDefaultFont",12,"bold")).grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,12))
        ttk.Label(left,text="Step").grid(row=1,column=0,sticky="w",pady=5); self.step_cb=ttk.Combobox(left,textvariable=self.step,state="disabled",width=26); self.step_cb.grid(row=1,column=1); self.step_cb.bind("<<ComboboxSelected>>",lambda e:self.selection())
        ttk.Label(left,text="Layer").grid(row=2,column=0,sticky="w",pady=5); self.layer_cb=ttk.Combobox(left,textvariable=self.layer,state="disabled",width=26); self.layer_cb.grid(row=2,column=1); self.layer_cb.bind("<<ComboboxSelected>>",lambda e:self.selection())
        ttk.Label(left,text="출력 방식").grid(row=3,column=0,sticky="w",pady=5); self.mode_cb=ttk.Combobox(left,textvariable=self.mode,values=[MODE_DPI,MODE_AOI],state="readonly",width=18); self.mode_cb.grid(row=3,column=1,sticky="w"); self.mode_cb.bind("<<ComboboxSelected>>",lambda e:self._mode_changed())
        ttk.Label(left,text="출력 DPI").grid(row=4,column=0,sticky="w",pady=5); self.dpi_sp=ttk.Spinbox(left,from_=MIN_DPI,to=MAX_DPI,increment=50,textvariable=self.dpi,width=12); self.dpi_sp.grid(row=4,column=1,sticky="w")
        self.aoi=ttk.LabelFrame(left,text="AOI 해상도 설정",padding=8); self.aoi.grid(row=5,column=0,columnspan=2,sticky="ew",pady=(8,4))
        ttk.Label(self.aoi,text="설비 프로파일").grid(row=0,column=0,sticky="w"); self.profile_cb=ttk.Combobox(self.aoi,textvariable=self.profile,state="readonly",width=20); self.profile_cb.grid(row=0,column=1); self.profile_cb.bind("<<ComboboxSelected>>",lambda e:self.use_profile())
        ttk.Label(self.aoi,text="X (µm/pixel)").grid(row=1,column=0,sticky="w",pady=3); self.xsp=ttk.Spinbox(self.aoi,from_=MIN_UM,to=MAX_UM,increment=.1,textvariable=self.umx,width=12); self.xsp.grid(row=1,column=1,sticky="w")
        ttk.Label(self.aoi,text="Y (µm/pixel)").grid(row=2,column=0,sticky="w",pady=3); self.ysp=ttk.Spinbox(self.aoi,from_=MIN_UM,to=MAX_UM,increment=.1,textvariable=self.umy,width=12); self.ysp.grid(row=2,column=1,sticky="w")
        ttk.Checkbutton(self.aoi,text="X/Y 동일 해상도",variable=self.lock,command=self.lock_changed).grid(row=3,column=0,columnspan=2,sticky="w"); ttk.Button(self.aoi,text="현재 값 프로파일 저장",command=self.save_profile).grid(row=4,column=0,columnspan=2,sticky="ew",pady=(5,0)); self.umx.trace_add("write",lambda *_:self.sync_y()); self.refresh_profiles()
        self.render_btn=ttk.Button(left,text="렌더링 결과 저장",command=self.render,state="disabled"); self.render_btn.grid(row=6,column=0,columnspan=2,sticky="ew",pady=(14,12)); ttk.Separator(left).grid(row=7,column=0,columnspan=2,sticky="ew")
        ttk.Label(left,text="ODB 정보",font=("TkDefaultFont",10,"bold")).grid(row=8,column=0,columnspan=2,sticky="w",pady=(8,4)); self.info_txt=tk.Text(left,width=40,height=18,wrap="word",state="disabled",relief="flat"); self.info_txt.grid(row=9,column=0,columnspan=2,sticky="nsew"); left.rowconfigure(9,weight=1)
        hdr=ttk.Frame(right); hdr.pack(fill="x",pady=(0,8)); ttk.Label(hdr,text="CAM 미리보기",font=("TkDefaultFont",12,"bold")).pack(side="left"); tools=ttk.Frame(hdr); tools.pack(side="right"); ttk.Button(tools,text="−",width=3,command=lambda:self.change_zoom(1/ZOOM_STEP)).pack(side="left"); ttk.Label(tools,textvariable=self.zoom_text,width=7,anchor="center").pack(side="left"); ttk.Button(tools,text="+",width=3,command=lambda:self.change_zoom(ZOOM_STEP)).pack(side="left"); ttk.Button(tools,text="맞춤",command=self.fit).pack(side="left",padx=(6,0)); ttk.Button(tools,text="100%",command=self.actual).pack(side="left",padx=(4,0))
        cf=ttk.Frame(right); cf.pack(fill="both",expand=True); cf.rowconfigure(0,weight=1); cf.columnconfigure(0,weight=1); self.canvas=tk.Canvas(cf,background="#202020",highlightthickness=0,cursor="fleur"); self.canvas.grid(row=0,column=0,sticky="nsew"); hs=ttk.Scrollbar(cf,orient="horizontal",command=self.canvas.xview); vs=ttk.Scrollbar(cf,orient="vertical",command=self.canvas.yview); hs.grid(row=1,column=0,sticky="ew"); vs.grid(row=0,column=1,sticky="ns"); self.canvas.configure(xscrollcommand=hs.set,yscrollcommand=vs.set)
        self.canvas.bind("<Configure>",lambda e:self.draw()); self.canvas.bind("<MouseWheel>",self.wheel); self.canvas.bind("<Button-4>",self.wheel); self.canvas.bind("<Button-5>",self.wheel); self.canvas.bind("<ButtonPress-1>",self.pan_start); self.canvas.bind("<B1-Motion>",self.pan_move); self.canvas.bind("<ButtonRelease-1>",self.pan_end); ttk.Label(root,textvariable=self.status,anchor="w",relief="sunken").pack(fill="x",pady=(10,0))

    def refresh_profiles(self):
        vals=[CUSTOM]+sorted(self.profiles); self.profile_cb["values"]=vals
        if self.profile.get() not in vals:self.profile.set(CUSTOM)
    def valid_um(self):
        try:x,y=float(self.umx.get()),float(self.umy.get())
        except Exception:raise ValueError("AOI 해상도는 숫자여야 합니다.")
        if not(MIN_UM<=x<=MAX_UM and MIN_UM<=y<=MAX_UM):raise ValueError(f"AOI 해상도는 {MIN_UM}~{MAX_UM} µm/pixel 범위여야 합니다.")
        return x,y
    def sync_y(self):
        if self.lock.get():
            try:self.umy.set(self.umx.get())
            except tk.TclError:pass
        self.update_info()
    def lock_changed(self):
        if self.lock.get():self.umy.set(self.umx.get());self.ysp.configure(state="disabled")
        elif self.mode.get()==MODE_AOI:self.ysp.configure(state="normal")
        self.update_info()
    def _mode_changed(self):
        a=self.mode.get()==MODE_AOI; self.dpi_sp.configure(state="disabled" if a else "normal")
        for w in self.aoi.winfo_children():
            try:w.configure(state="normal" if a else "disabled")
            except tk.TclError:pass
        if a:self.profile_cb.configure(state="readonly");self.lock_changed()
        self.update_info()
    def save_profile(self):
        try:x,y=self.valid_um()
        except ValueError as e:return messagebox.showerror("AOI 해상도 오류",str(e))
        name=simpledialog.askstring("설비 프로파일 저장","설비/프로파일 이름을 입력하세요.",parent=self)
        if not name:return
        self.profiles[name.strip()]={"um_per_pixel_x":x,"um_per_pixel_y":y};save_profiles(self.profiles);self.refresh_profiles();self.profile.set(name.strip());self.status.set(f"프로파일 저장: {name.strip()}")
    def use_profile(self):
        p=self.profiles.get(self.profile.get())
        if p:self.lock.set(abs(float(p["um_per_pixel_x"])-float(p.get("um_per_pixel_y",p["um_per_pixel_x"])))<1e-12);self.umx.set(float(p["um_per_pixel_x"]));self.umy.set(float(p.get("um_per_pixel_y",p["um_per_pixel_x"])));self.lock_changed()

    def preview_settings(self):
        d=tk.Toplevel(self);d.title("설정");d.transient(self);d.grab_set();f=ttk.Frame(d,padding=18);f.pack();ttk.Label(f,text="미리보기 DPI").grid(row=0,column=0,padx=(0,16));v=tk.StringVar(value=str(self.preview_dpi));sp=ttk.Spinbox(f,from_=MIN_DPI,to=MAX_DPI,increment=50,textvariable=v,width=12);sp.grid(row=0,column=1)
        def apply():
            try:n=int(v.get());assert MIN_DPI<=n<=MAX_DPI
            except Exception:return messagebox.showerror("DPI 오류",f"{MIN_DPI}~{MAX_DPI} 범위의 정수를 입력하세요.",parent=d)
            changed=n!=self.preview_dpi;self.preview_dpi=n;d.destroy();self.update_info();self.schedule_preview() if changed and self.job else None
        b=ttk.Frame(f);b.grid(row=1,column=0,columnspan=2,sticky="e",pady=(14,0));ttk.Button(b,text="취소",command=d.destroy).pack(side="right");ttk.Button(b,text="적용",command=apply).pack(side="right",padx=6)

    def open_file(self):
        p=filedialog.askopenfilename(title="ODB++ TGZ 파일 선택",filetypes=[("ODB++ TGZ","*.tgz"),("Tar GZip","*.tar.gz"),("모든 파일","*.*")],defaultextension=".tgz")
        if p:self.load(Path(p))
    def load(self,path:Path):
        self.status.set("ODB++ 파일을 분석하는 중...");self.token+=1;t=self.token
        def work():
            try:j,tmp=extract_input(path);self.q.put(("loaded",t,path,j,tmp,inspect_job(j)))
            except Exception as e:self.q.put(("error",t,f"파일 로드 실패: {e}"))
        threading.Thread(target=work,daemon=True).start()
    def apply_loaded(self,path,j,tmp,info):
        if self.tmp:self.tmp.cleanup()
        self.job,self.tmp,self.info,self.source=j,tmp,info,path;self.file_label.configure(text=path.name);self.step_cb["values"]=info.steps;self.step.set("unit" if "unit" in info.steps else info.steps[0]);self.step_cb.configure(state="readonly")
        labels=[x.label for x in info.layers];self.layer_cb["values"]=labels;idx=next((i for i,x in enumerate(info.layers) if x.name.lower()=="l1"),0);self.layer_cb.current(idx);self.layer_cb.configure(state="readonly");self.render_btn.configure(state="normal");self.update_info();self.schedule_preview()
    def selected_layer(self):return next((x for x in self.info.layers if x.label==self.layer.get()),None) if self.info else None
    def selection(self):self.update_info();self.schedule_preview()
    def schedule_preview(self):
        if not self.job:return
        l=self.selected_layer();s=self.step.get()
        if not l or not s:return
        self.token+=1;t=self.token;dpi=self.preview_dpi;self.render_btn.configure(state="disabled");self.status.set(f"미리보기 생성 중: {s}/{l.name} @ {dpi} DPI")
        def work():
            try:r=ODBRenderer(self.job,dpi);im=r.render(s,l.name);self.q.put(("preview",t,im,r.stats,dpi))
            except Exception as e:self.q.put(("error",t,f"미리보기 생성 실패: {e}"))
        threading.Thread(target=work,daemon=True).start()
    def poll(self):
        try:
            while True:
                r=self.q.get_nowait();k=r[0]
                if k=="loaded":
                    _,t,p,j,tmp,info=r
                    if t==self.token:self.apply_loaded(p,j,tmp,info)
                    elif tmp:tmp.cleanup()
                elif k=="preview":
                    _,t,im,st,dpi=r
                    if t==self.token:self.preview=im;self.fit();self.render_btn.configure(state="normal");self.status.set(f"미리보기 완료 | {im.width}×{im.height}px @ {dpi} DPI | warnings={st.unsupported}")
                elif k=="rendered":
                    _,im,out,st,desc=r;self.render_btn.configure(state="normal");self.status.set(f"렌더링 완료: {out}");messagebox.showinfo("렌더링 완료",f"{out}\n\n{desc}\n크기: {im.width} × {im.height}px\nWarnings: {st.unsupported}")
                elif k=="error":
                    _,t,msg=r
                    if t==self.token:self.render_btn.configure(state="normal" if self.job else "disabled");self.status.set(msg);messagebox.showerror("오류",msg)
        except queue.Empty:pass
        self.after(100,self.poll)

    def fit(self):
        if not self.preview:return self.draw()
        self.zoom=max(ZOOM_MIN,min(ZOOM_MAX,min(max(1,self.canvas.winfo_width()-32)/self.preview.width,max(1,self.canvas.winfo_height()-32)/self.preview.height)));self.offx=self.offy=0.;self.draw()
    def actual(self):self.zoom=1.;self.offx=self.offy=0.;self.draw()
    def change_zoom(self,f,x=None,y=None):
        if not self.preview:return
        old=self.zoom;new=max(ZOOM_MIN,min(ZOOM_MAX,old*f));cw,ch=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height());ax=cw/2 if x is None else x;ay=ch/2 if y is None else y;ix=(ax-cw/2-self.offx)/old;iy=(ay-ch/2-self.offy)/old;self.zoom=new;self.offx=ax-cw/2-ix*new;self.offy=ay-ch/2-iy*new;self.draw()
    def wheel(self,e):self.change_zoom(ZOOM_STEP if getattr(e,"num",None)==4 or getattr(e,"delta",0)>0 else 1/ZOOM_STEP,e.x,e.y);return "break"
    def pan_start(self,e):self.pan=(e.x,e.y)
    def pan_move(self,e):
        if not self.pan:return
        x,y=self.pan;self.offx+=e.x-x;self.offy+=e.y-y;self.pan=(e.x,e.y);self.draw()
    def pan_end(self,e):self.pan=None
    def draw(self):
        self.canvas.delete("all")
        if not self.preview:return self.canvas.create_text(max(10,self.canvas.winfo_width()//2),max(10,self.canvas.winfo_height()//2),text="ODB++ 파일을 열고 Layer를 선택하면 미리보기가 표시됩니다.",fill="white")
        w=max(1,round(self.preview.width*self.zoom));h=max(1,round(self.preview.height*self.zoom));rs=Image.Resampling.NEAREST if self.zoom>=1 else Image.Resampling.LANCZOS;im=self.preview.resize((w,h),rs);self.photo=ImageTk.PhotoImage(ImageOps.colorize(im,black="black",white="white"));cw,ch=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height());self.canvas.create_image(cw/2+self.offx,ch/2+self.offy,image=self.photo,anchor="center",tags="p");b=self.canvas.bbox("p");self.canvas.configure(scrollregion=(min(0,b[0]-100),min(0,b[1]-100),max(cw,b[2]+100),max(ch,b[3]+100)));self.zoom_text.set(f"{self.zoom*100:.0f}%")

    def update_info(self):
        if not hasattr(self,"info_txt") or not self.job or not self.info:return
        l=self.selected_layer();s=self.step.get();lines=[f"Job: {self.info.name}",f"Source: {self.source.name}",f"Step: {s}"];wh=None
        try:wh=profile_size_mm(self.job,s);lines.append(f"Profile: {wh[0]:.3f} × {wh[1]:.3f} mm")
        except Exception:pass
        if l:
            lines += ["",f"Layer: {l.name}",f"Type: {l.layer_type}",f"Side: {l.side}","",f"Preview DPI: {self.preview_dpi}",f"Output Mode: {self.mode.get()}"]
            if self.mode.get()==MODE_DPI:lines.append(f"Output DPI: {self.dpi.get()}")
            else:
                try:x,y=self.valid_um();lines += [f"AOI X: {x:g} µm/pixel",f"AOI Y: {y:g} µm/pixel",f"Equivalent DPI: X={25400/x:.2f}, Y={25400/y:.2f}"];lines.append(f"Expected pixels: ≈ {math.ceil(wh[0]*1000/x)} × {math.ceil(wh[1]*1000/y)}") if wh else None
                except ValueError:lines.append("AOI resolution: invalid")
        self.info_txt.configure(state="normal");self.info_txt.delete("1.0","end");self.info_txt.insert("1.0","\n".join(lines));self.info_txt.configure(state="disabled")

    def render(self):
        if not self.job:return
        l=self.selected_layer();s=self.step.get()
        if not l:return
        try:
            if self.mode.get()==MODE_DPI:
                dpi=int(self.dpi.get());assert MIN_DPI<=dpi<=MAX_DPI;args=("dpi",dpi,dpi);suffix=f"{dpi}dpi";desc=f"DPI: {dpi}"
            else:
                x,y=self.valid_um();args=("aoi",x,y);suffix=f"{x:g}x{y:g}umpp".replace(".","p");desc=f"AOI 해상도: X={x:g}, Y={y:g} µm/pixel"
        except Exception as e:return messagebox.showerror("출력 설정 오류",str(e) or "출력 설정값을 확인하세요.")
        out=filedialog.asksaveasfilename(title="CAM Image 저장",defaultextension=".png",initialfile=f"{self.info.name}_{s}_{l.name}_{suffix}.png",filetypes=[("PNG Image","*.png")])
        if not out:return
        out=Path(out);t=self.token;self.render_btn.configure(state="disabled");self.status.set(f"렌더링 중: {s}/{l.name} | {desc}")
        def work():
            try:
                r=ODBRenderer(self.job,args[1]) if args[0]=="dpi" else ODBRenderer.from_um_per_pixel(self.job,args[1],args[2]);im=r.render(s,l.name);out.parent.mkdir(parents=True,exist_ok=True);im.save(out,optimize=True,dpi=(r.dpi_x,r.dpi_y));self.q.put(("rendered",im,out,r.stats,desc))
            except Exception as e:self.q.put(("error",t,f"렌더링 실패: {e}"))
        threading.Thread(target=work,daemon=True).start()
    def close(self):
        if self.tmp:self.tmp.cleanup()
        self.destroy()

if __name__=="__main__":App().mainloop()
