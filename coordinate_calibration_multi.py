#!/usr/bin/env python3
from __future__ import annotations

import math, queue, threading, tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional
from PIL import Image, ImageOps, ImageTk

from app import inspect_job, render_layer
from coordinate_transform import CalibrationPoint, extract_xy_from_filename, fit_affine
from odb_cam_renderer import extract_input

DPI=600; MIN_ZOOM=.05; MAX_ZOOM=16.; ZOOM_STEP=1.25

@dataclass
class Crop:
    path: Path
    ax: Optional[float]
    ay: Optional[float]
    ox: Optional[float]=None
    oy: Optional[float]=None
    err: Optional[float]=None
    @property
    def matched(self): return None not in (self.ax,self.ay,self.ox,self.oy)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AOI ↔ ODB Multi-Crop Calibration"); self.geometry("1500x900")
        self.job_dir=self.job_tmp=self.job_info=self.preview=self.preview_photo=None
        self.source_file=None; self.crops=[]; self.idx=None; self.crop_photo=None; self.transform=None
        self.zoom=1.; self.offx=self.offy=0.; self.pan=None; self.pick=False; self.marker=None
        self.q=queue.Queue(); self.token=0
        self.step=tk.StringVar(); self.layer=tk.StringVar(); self.dpi=tk.IntVar(value=DPI)
        self.ax=tk.StringVar(); self.ay=tk.StringVar()
        self.status=tk.StringVar(value="ODB++ 파일을 열어주세요.")
        self.progress=tk.StringVar(value="Crop 0개 / 매칭 0개")
        self.result=tk.StringVar(value="최소 3개 Crop을 ODB 위치와 매칭하세요.")
        self.pick_text=tk.StringVar(value="현재 Crop ODB 위치 선택")
        self.build(); self.protocol("WM_DELETE_WINDOW",self.close); self.after(100,self.poll)

    def build(self):
        root=ttk.Frame(self,padding=10); root.pack(fill="both",expand=True)
        top=ttk.Frame(root); top.pack(fill="x",pady=(0,8))
        ttk.Button(top,text="ODB++ 파일 열기",command=self.open_odb).pack(side="left")
        self.file_label=ttk.Label(top,text="선택된 ODB 없음"); self.file_label.pack(side="left",padx=10)
        for text,var,width in (("Step",self.step,16),("Layer",self.layer,26)):
            ttk.Label(top,text=text).pack(side="left",padx=(10,4))
            cb=ttk.Combobox(top,textvariable=var,state="disabled",width=width); cb.pack(side="left")
            cb.bind("<<ComboboxSelected>>",lambda e:self.render())
            if text=="Step": self.step_cb=cb
            else: self.layer_cb=cb
        ttk.Label(top,text="DPI").pack(side="left",padx=(10,4))
        ttk.Spinbox(top,from_=72,to=10000,increment=50,textvariable=self.dpi,width=8).pack(side="left")
        ttk.Button(top,text="미리보기 갱신",command=self.render).pack(side="left",padx=6)

        pane=ttk.Panedwindow(root,orient="horizontal"); pane.pack(fill="both",expand=True)
        left=ttk.Frame(pane,padding=4); right=ttk.Frame(pane,padding=4)
        pane.add(left,weight=0); pane.add(right,weight=1); left.columnconfigure(0,weight=1)

        ttk.Label(left,text="AOI Crop 목록",font=("TkDefaultFont",11,"bold")).grid(row=0,column=0,sticky="w")
        b=ttk.Frame(left); b.grid(row=1,column=0,sticky="ew",pady=5)
        ttk.Button(b,text="여러 Crop 불러오기",command=self.open_crops).pack(side="left",fill="x",expand=True)
        ttk.Button(b,text="초기화",command=self.clear).pack(side="left",padx=(4,0))
        self.tree=ttk.Treeview(left,columns=("s","a","o","e"),show="tree headings",height=12,selectmode="browse")
        for c,t,w in (("#0","Crop",200),("s","상태",45),("a","AOI X,Y",125),("o","ODB X,Y",125),("e","Err",60)):
            self.tree.heading(c,text=t); self.tree.column(c,width=w,anchor="center" if c!="#0" else "w")
        self.tree.grid(row=2,column=0,sticky="nsew"); left.rowconfigure(2,weight=1)
        self.tree.bind("<<TreeviewSelect>>",self.tree_select)
        ttk.Label(left,textvariable=self.progress).grid(row=3,column=0,sticky="w",pady=(4,8))

        box=ttk.LabelFrame(left,text="선택 Crop",padding=8); box.grid(row=4,column=0,sticky="ew")
        self.crop_label=ttk.Label(box,text="Crop을 선택하세요.",anchor="center"); self.crop_label.grid(row=0,column=0,columnspan=4,sticky="ew")
        ttk.Label(box,text="AOI X").grid(row=1,column=0); ttk.Entry(box,textvariable=self.ax,width=12).grid(row=1,column=1,padx=4)
        ttk.Label(box,text="AOI Y").grid(row=1,column=2); ttk.Entry(box,textvariable=self.ay,width=12).grid(row=1,column=3,padx=4)
        ttk.Button(box,text="좌표 적용",command=self.apply_xy).grid(row=2,column=0,columnspan=4,sticky="ew",pady=(6,0))

        ttk.Button(left,textvariable=self.pick_text,command=self.arm_pick).grid(row=5,column=0,sticky="ew",pady=(10,4))
        ttk.Button(left,text="현재 Crop 대응점 등록",command=self.register).grid(row=6,column=0,sticky="ew")
        ttk.Button(left,text="Calibration 계산",command=self.calibrate).grid(row=7,column=0,sticky="ew",pady=(8,4))
        out=ttk.LabelFrame(left,text="Calibration 결과",padding=8); out.grid(row=8,column=0,sticky="ew")
        ttk.Label(out,textvariable=self.result,wraplength=390,justify="left").pack(fill="x")

        ttk.Label(right,text="ODB CAM 미리보기",font=("TkDefaultFont",11,"bold")).pack(anchor="w")
        self.canvas=tk.Canvas(right,bg="#202020",highlightthickness=0,cursor="fleur"); self.canvas.pack(fill="both",expand=True,pady=(6,0))
        self.canvas.bind("<Configure>",lambda e:self.draw()); self.canvas.bind("<MouseWheel>",self.wheel)
        self.canvas.bind("<Button-4>",self.wheel); self.canvas.bind("<Button-5>",self.wheel)
        self.canvas.bind("<ButtonPress-1>",self.press); self.canvas.bind("<B1-Motion>",self.drag); self.canvas.bind("<ButtonRelease-1>",lambda e:setattr(self,"pan",None))
        ttk.Label(root,textvariable=self.status,anchor="w",relief="sunken").pack(fill="x",pady=(8,0))

    def open_odb(self):
        p=filedialog.askopenfilename(filetypes=[("ODB++ TGZ","*.tgz"),("Tar GZip","*.tar.gz"),("All","*.*")])
        if not p:return
        self.token+=1; token=self.token; self.status.set("ODB 분석 중...")
        def worker():
            try:
                jd,tmp=extract_input(Path(p)); self.q.put(("load",token,Path(p),jd,tmp,inspect_job(jd)))
            except Exception as e:self.q.put(("err",token,str(e)))
        threading.Thread(target=worker,daemon=True).start()

    def apply_job(self,p,jd,tmp,info):
        if self.job_tmp:self.job_tmp.cleanup()
        self.source_file,self.job_dir,self.job_tmp,self.job_info=p,jd,tmp,info; self.file_label.config(text=p.name)
        self.step_cb["values"]=info.steps
        pref=next((x for x in ("unit","strip","pnl") if x in info.steps),info.steps[0] if info.steps else "")
        self.step.set(pref); self.step_cb.config(state="readonly" if info.steps else "disabled")
        vals=[x.label for x in info.layers]; self.layer_cb["values"]=vals
        if vals:self.layer_cb.current(0)
        self.layer_cb.config(state="readonly" if vals else "disabled")
        if pref and vals:self.render()

    def layer_name(self):
        if not self.job_info:return None
        return next((x.name for x in self.job_info.layers if x.label==self.layer.get()),None)

    def render(self):
        if not self.job_dir:return
        step,layer=self.step.get(),self.layer_name()
        if not step or not layer:return
        try:dpi=int(self.dpi.get())
        except: return messagebox.showerror("DPI 오류","DPI는 정수여야 합니다.")
        self.token+=1; token=self.token; jd=self.job_dir; self.status.set("ODB 미리보기 생성 중...")
        def worker():
            try:self.q.put(("preview",token,render_layer(jd,step,layer,dpi)[0]))
            except Exception as e:self.q.put(("err",token,str(e)))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        try:
            while True:
                r=self.q.get_nowait()
                if r[0]=="load":
                    _,t,p,jd,tmp,info=r
                    if t==self.token:self.apply_job(p,jd,tmp,info)
                    else:tmp.cleanup()
                elif r[0]=="preview" and r[1]==self.token:
                    self.preview=r[2]; self.marker=None; self.fit(); self.status.set(f"미리보기 완료: {self.preview.width}×{self.preview.height}px")
                elif r[0]=="err" and r[1]==self.token: messagebox.showerror("오류",r[2])
        except queue.Empty:pass
        self.after(100,self.poll)

    def open_crops(self):
        ps=filedialog.askopenfilenames(title="동일 Unit AOI Crop 선택",filetypes=[("Image","*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),("All","*.*")])
        for s in ps:
            p=Path(s); xy=extract_xy_from_filename(p.name); self.crops.append(Crop(p,xy[0] if xy else None,xy[1] if xy else None))
        self.refresh()
        if self.idx is None and self.crops:self.select(0)

    def clear(self):
        self.crops=[]; self.idx=None; self.transform=None; self.crop_photo=None; self.marker=None
        self.crop_label.config(image="",text="Crop을 선택하세요."); self.ax.set(""); self.ay.set("")
        self.result.set("최소 3개 Crop을 ODB 위치와 매칭하세요."); self.refresh(); self.draw()

    def refresh(self):
        cur=self.idx
        for i in self.tree.get_children():self.tree.delete(i)
        for i,c in enumerate(self.crops):
            self.tree.insert("", "end", iid=str(i), text=c.path.name, values=("✓" if c.matched else "·", "-" if c.ax is None else f"{c.ax:g},{c.ay:g}", "-" if c.ox is None else f"{c.ox:.1f},{c.oy:.1f}", "-" if c.err is None else f"{c.err:.2f}"))
        self.progress.set(f"Crop {len(self.crops)}개 / 매칭 {sum(c.matched for c in self.crops)}개")
        if cur is not None and cur<len(self.crops): self.tree.selection_set(str(cur))

    def tree_select(self,e=None):
        s=self.tree.selection()
        if s:self.select(int(s[0]))

    def select(self,i):
        self.idx=i; c=self.crops[i]
        try:
            im=ImageOps.grayscale(Image.open(c.path)); im.thumbnail((390,220),Image.Resampling.LANCZOS)
            self.crop_photo=ImageTk.PhotoImage(im); self.crop_label.config(image=self.crop_photo,text="")
        except Exception as e:self.crop_label.config(image="",text=str(e))
        self.ax.set("" if c.ax is None else f"{c.ax:g}"); self.ay.set("" if c.ay is None else f"{c.ay:g}")
        self.marker=(c.ox,c.oy) if c.ox is not None else None; self.draw()

    def apply_xy(self):
        if self.idx is None:return
        try:self.crops[self.idx].ax=float(self.ax.get()); self.crops[self.idx].ay=float(self.ay.get())
        except:return messagebox.showerror("좌표 오류","AOI X/Y를 확인하세요.")
        self.transform=None; self.refresh()

    def arm_pick(self):
        if self.preview is None or self.idx is None:return messagebox.showwarning("선택 필요","ODB 미리보기와 Crop을 먼저 선택하세요.")
        self.pick=True; self.pick_text.set("ODB 화면에서 동일 위치 클릭"); self.canvas.config(cursor="crosshair")

    def register(self):
        if self.idx is None:return
        c=self.crops[self.idx]
        try:c.ax=float(self.ax.get()); c.ay=float(self.ay.get())
        except:return messagebox.showerror("좌표 오류","AOI X/Y를 확인하세요.")
        if c.ox is None:return messagebox.showwarning("ODB 좌표 필요","ODB 위치를 먼저 선택하세요.")
        self.transform=None; c.err=None; self.refresh(); self.next_unmatched()

    def next_unmatched(self):
        start=(self.idx or 0)+1
        for k in range(len(self.crops)):
            i=(start+k)%len(self.crops)
            if not self.crops[i].matched:
                self.select(i); self.tree.selection_set(str(i)); self.tree.see(str(i)); return

    def calibrate(self):
        m=[c for c in self.crops if c.matched]
        if len(m)<3:return messagebox.showwarning("대응점 부족",f"최소 3개 필요, 현재 {len(m)}개")
        try:self.transform=fit_affine([CalibrationPoint(c.ax,c.ay,c.ox,c.oy) for c in m])
        except ValueError as e:return messagebox.showerror("Calibration 실패",str(e))
        for c in m:
            px,py=self.transform.apply(c.ax,c.ay); c.err=math.hypot(px-c.ox,py-c.oy)
        t=self.transform; self.result.set(f"ODB_X = {t.a:.8g}·AOI_X + {t.b:.8g}·AOI_Y + {t.c:.8g}\nODB_Y = {t.d:.8g}·AOI_X + {t.e:.8g}·AOI_Y + {t.f:.8g}\nRMSE = {t.rmse:.3f}px / {t.point_count}개")
        self.refresh(); self.status.set(f"Calibration 완료 / RMSE {t.rmse:.3f}px")

    def image_xy(self,x,y):
        if self.preview is None:return None
        cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height(); cx,cy=cw/2+self.offx,ch/2+self.offy
        ix=(x-cx)/self.zoom+self.preview.width/2; iy=(y-cy)/self.zoom+self.preview.height/2
        return (ix,iy) if 0<=ix<self.preview.width and 0<=iy<self.preview.height else None

    def press(self,e):
        if self.pick:
            p=self.image_xy(e.x,e.y)
            if p and self.idx is not None:
                c=self.crops[self.idx]; c.ox,c.oy=p; c.err=None; self.transform=None; self.marker=p; self.refresh(); self.tree.selection_set(str(self.idx))
                self.pick=False; self.pick_text.set("현재 Crop ODB 위치 선택"); self.canvas.config(cursor="fleur"); self.draw()
            return
        self.pan=(e.x,e.y)

    def drag(self,e):
        if self.pan and not self.pick:
            x,y=self.pan; self.offx+=e.x-x; self.offy+=e.y-y; self.pan=(e.x,e.y); self.draw()

    def wheel(self,e):
        if self.preview is None:return "break"
        f=ZOOM_STEP if getattr(e,"num",None)==4 or getattr(e,"delta",0)>0 else 1/ZOOM_STEP
        old=self.zoom; new=max(MIN_ZOOM,min(MAX_ZOOM,old*f))
        cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height()
        ix=(e.x-cw/2-self.offx)/old; iy=(e.y-ch/2-self.offy)/old
        self.zoom=new; self.offx=e.x-cw/2-ix*new; self.offy=e.y-ch/2-iy*new; self.draw(); return "break"

    def fit(self):
        if self.preview is None:return
        self.zoom=max(MIN_ZOOM,min(MAX_ZOOM,min((self.canvas.winfo_width()-30)/self.preview.width,(self.canvas.winfo_height()-30)/self.preview.height)))
        self.offx=self.offy=0.; self.draw()

    def draw(self):
        self.canvas.delete("all")
        if self.preview is None:return
        w=max(1,int(self.preview.width*self.zoom)); h=max(1,int(self.preview.height*self.zoom))
        im=self.preview.resize((w,h),Image.Resampling.NEAREST if self.zoom>=1 else Image.Resampling.LANCZOS)
        self.preview_photo=ImageTk.PhotoImage(ImageOps.colorize(im,black="black",white="white"))
        cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height(); cx,cy=cw/2+self.offx,ch/2+self.offy
        self.canvas.create_image(cx,cy,image=self.preview_photo)
        if self.marker:
            mx=cx+(self.marker[0]-self.preview.width/2)*self.zoom; my=cy+(self.marker[1]-self.preview.height/2)*self.zoom
            self.canvas.create_line(mx-12,my,mx+12,my,fill="#ff4040",width=2); self.canvas.create_line(mx,my-12,mx,my+12,fill="#ff4040",width=2)

    def close(self):
        if self.job_tmp:self.job_tmp.cleanup()
        self.destroy()

if __name__=="__main__": App().mainloop()
