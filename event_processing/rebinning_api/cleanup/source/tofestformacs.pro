;+
; NAME:
; TofEstForMACS
;
; PURPOSE:
; Simple program to estimate the average neutron paths and time-of-flight
; on MACS.
;
; This is done by generating random points within the components along
; the flight path (sample, analyzers and detector) and then calculating
; the total distances between these points and then making a statistical
; analysis on these to determine their mean and standard deviation in
; order to calculate the mean neutron tof and standard deviation. This
; estimate is simplistic and only takes into account the physical extent
; of the components and nothing else.
;
; KEYWORDS:
;  w_sample - width of sample (same as diameter if assume a cylindrical shape)
;
;  h_sample - height of sample
;
;  noplot   - set to disable plotting the results
;
;  nPoints  - the number of points to generate within each of the components
;
;  AUTHORS:
;  Yiming Qiu
;  Richard Azuah
;  July, 2021
;-
pro TofEstForMACS,Ef,w_sample=w_sample,h_sample=h_sample, $
  nPoints=nPoints, noplot=noplot,dx=d_x,dy=d_y,dz=d_z
  if N_elements(Ef) eq 0 then Ef = 5.0  ; default 5.0 meV
  if N_elements(w_sample) eq 0 then w = 1/100. else w = w_sample/100.  ;cm->m
  if N_elements(h_sample) eq 0 then h = 2/100. else h = h_sample/100.  ;cm->m
  if N_elements(nPoints) eq 0 then nPoints = 31 

  Lsa = 1.00  ;sample to ananlyzer distance in m when a6=90
  Laa = 0.07  ;analyzer to analyzer distance in m when a6=90
  Lad = 0.25  ;analyzer to detector distance in m when a6=90
  a6 = Asin(Sqrt(81.8042/Ef)/2.0/3.35416)*2 ;Ef in meVTofEstForC
  v  = Sqrt(Ef/5.22704e-6) ;neutron velocity in m/s from Ef in meV
  dEf = -0.188852+0.107677*Ef   ;empirical formula for Ef<5 meV with Be filter
  dv = v/(2.*Ef)*dEf
  L0 = (Lsa+Lad+Laa/Sin(a6)-Laa/Tan(a6))  ;travel distance for beam center
  time = L0/v  ;in seconds

  ;######################################################################
  ; Based on the geometry of the instrument, determine (min, max) limits along the x, y and z axes
  ; for each of the components
  ; Then, generate random points that lie within the set limits.

  ;analyzer dimension 6 cm (width) x 2cm (ignore 2mm thickness) x 9, in curvature of 50 cm
  w_a = 0.06                      ;in m
  h_a = 0.5*Sin(Asin(1./50)*9)*2  ;~0.18 m
  ;sample coordinates, assuming cylindrical shape
  s_x_min = -w/2.0 & s_x_max = w/2.0
  s_y_min = s_x_min & s_y_max = s_x_max
  s_z_min = -h/2.0 & s_z_max = h/2.0
  ns = nPoints
  seed = !NULL
  s_x = Randomu(seed, ns) * (s_x_max - s_x_min) + s_x_min
  seed = !NULL
  s_y = Randomu(seed, ns) * (s_y_max - s_y_min) + s_y_min
  seed = !NULL
  s_z = Randomu(seed, ns) * (s_z_max - s_z_min) + s_z_min

  ;analyzer one coordinates
  a1_x_min = -w_a/2.0*Sin(a6/2.)
  a1_x_max = w_a/2.0*Sin(a6/2.)
  a1_y_min = Lsa-Laa/Tan(a6)/2. - w_a/2.*Cos(a6/2.)
  a1_y_max = Lsa-Laa/Tan(a6)/2. + w_a/2.0*Cos(a6/2.)
  a1_z_min = -h_a/2.0
  a1_z_max = h_a/2.0
  na = nPoints
  seed = !NULL
  a1_x = Randomu(seed, na) * (a1_x_max - a1_x_min) + a1_x_min
  seed = !NULL
  a1_y = Randomu(seed, na) * (a1_y_max - a1_y_min) + a1_y_min
  seed = !NULL
  a1_z = Randomu(seed, na) * (a1_z_max - a1_z_min) + a1_z_min

  ;analyzer two coordinates
  a2_x_min = Laa - w_a/2.0*Sin(a6/2.)
  a2_x_max = Laa + w_a/2.0*Sin(a6/2.)
  a2_y_min = Lsa+Laa/Tan(a6)/2. - w_a/2.*Cos(a6/2.)
  a2_y_max = Lsa+Laa/Tan(a6)/2. + w_a/2.0*Cos(a6/2.)
  a2_z_min = a1_z_min
  a2_z_max = a1_z_max
  seed = !NULL
  a2_x = Randomu(seed, na) * (a2_x_max - a2_x_min) + a2_x_min
  seed = !NULL
  a2_y = Randomu(seed, na) * (a2_y_max - a2_y_min) + a2_y_min
  seed = !NULL
  a2_z = Randomu(seed, na) * (a2_z_max - a2_z_min) + a2_z_min

  ;detector coordinates, detecotr height 14 cm, diameter 2.6 cm
  h_d  = 0.14               ;in m
  r_d  = 0.013              ;in m

  d_x_min = Laa - r_d
  d_x_max = Laa + r_d
  d_y_min = Lsa + Lad - r_d
  d_y_max = Lsa + Lad + r_d
  d_z_min = -h_d/2.0
  d_z_max = h_d/2.0
  nd = nPoints
  seed = !NULL
  d_x = Randomu(seed, nd) * (d_x_max - d_x_min) + d_x_min
  seed = !NULL
  d_y = Randomu(seed, nd) * (d_y_max - d_y_min) + d_y_min
  seed = !NULL
  d_z = Randomu(seed, nd) * (d_z_max - d_z_min) + d_z_min
  
  

  Tic
  ;######################################################################
  ; Now calculate the distance between the components using the points generated above
  ; to determine the total sample-dector distance.
  ; Accumulate the result to obtain a random distribution of sample-detector distances
  Lsd = []
  for i=0,ns-1 do begin
    for j=0,na-1 do begin
      for k=0,na-1 do begin
        for l=0,nd-1 do begin
          L1=Sqrt((a1_x[j]-s_x[i])^2+(a1_y[j]-s_y[i])^2+(a1_z[j]-s_z[i])^2)+$
            Sqrt((a2_x[k]-a1_x[j])^2+(a2_y[k]-a1_y[j])^2+(a2_z[k]-a1_z[j])^2)+$
            Sqrt((d_x[l]-a2_x[k])^2+(d_y[l]-a2_y[k])^2+(d_z[l]-a2_z[k])^2)
          Lsd = [Lsd,L1]
        endfor
      endfor
    endfor
  endfor

  Toc
  Print,N_elements(Lsd)

  dL = Max(Abs(Lsd-L0))
  d_time = time*Sqrt((dL/L0)^2+(dv/v)^2)
  Print,'Ef =',Ef,' meV, time =',time,' +/-',d_time,' secs'


  ;######################################################################
  ; Use statistical analysis to determine mean and standard deviation of
  ; the distance and hence tof
  moments = Moment(Lsd)
  Print,''
  Print,'##################################################'
  Print,'*** MACS ***
  Print,'##################################################'
  fmt1 = '(F6.1)'
  fmt2 = '(F6.3)'
  results='Ef = '+String(Ef,format=fmt)+' meV'
  results = [results,'Mean S-D dist = '+String(moments[0]*1000.0,format=fmt1)+' mm']
  results = [results,'$\sigma$ of S-D dist = '+String(Sqrt(moments[1])*1000.0,format=fmt1)+' mm']
  results = [results,'Mean TOF      = '+String(moments[0]/v*1000,format=fmt2)+' ms']
  results = [results,'$\sigma$ of TOF      = '+String(Sqrt(moments[1])/v*1000,format=fmt2)+' ms']
  results = [results,'$\Delta t/t$    = '+string(Sqrt(moments[1])/moments[0]*100, format='(F4.1)')+' %']
  for i=0,5 do Print,results[i]
  Print,'##################################################'

  if (Keyword_set(noplot)) then Return

  ;######################################################################
  ; Histogram the results to enable a plot of the distribution of distances and times
  frequency = Histogram(Lsd, nbins=51, locations=distances)
  times = distances/v
  xtitle='L$_{samp-detector}$ (m)'
  ytitle="'Intensity'"
  title='MACS: Lsd Estimates for Ef = '+String(Ef,format='(F5.2)')+' meV'
  p1 = Plot(distances,frequency,ytitle=ytitle,xtitle=xtitle,title=title, $
    dim=[1000,600],linestyle=' ',symbol='diamond',color=blue)
  xtitle='TOF$_{samp-detector} (s)$'
  title='MACS: TOF Estimates for Ef = '+String(Ef,format='(F5.2)')+' meV'
  p2 = Plot(times,frequency,ytitle=ytitle,xtitle=xtitle,title=title, $
    dim=[1000,600],linestyle=' ',symbol='diamond',color=blue)
  void = Text(0.5,0.7,results[[0,1,2]],/normal,target=p1,font_name='Courier')
  void = Text(0.5,0.7,results[[0,3,4,5]],/normal,target=p2,font_name='Courier')

end
