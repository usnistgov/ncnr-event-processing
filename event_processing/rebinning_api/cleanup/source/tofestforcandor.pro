;+
; NAME:
; TofEstForCANDOR
;
; PURPOSE:
; Simple program to estimate the average neutron paths and time-of-flight
; on CANDOR.
;
; This is done by generating random points within the components along
; the flight path (sample, HOPG analyzer and detector) and then calculating
; the total distances between these points and then making a statistical
; analysis on these to determine their mean and standard deviation in
; order to calculate the mean neutron tof and standard deviation. This
; estimate is simplistic and only takes into account the physical extent
; of the components and nothing else.
; 
; PARAMETERS
;  wavelength - neutron wavelength
;
; KEYWORDS:
;  SampleDim   - Sample dimensions [Length,Width,thickness] where Length is parallel to the beam in mm
;  
;  AnalyzerDim - HOPG dimensions [width, Height, thickness] in mm
;  
;  DetectorDim - Detector dimensions [width, Height, thickness] in mm
;  
;  thetaS      - the specular angle of the beam on the sample (for detector bank0) in degrees
;  
;  thetaA      - HOPG theta in degrees
;  
;  Lsa         - Sample-HOPG distance in mm
;  
;  Lad         - HOPG-Detector distance in mm
;  
;  nPoints     - the number of points to generate within each of the components
;  
;  noPlotFlag  - set to disable plotting of the results
;
; AUTHORS:
;  Richard Azuah
;  July, 2021
;-
pro TofEstForCANDOR              $
  ,wavelength               $  ; neutron wavelength
  ,DeltaLambda=DeltaLambda  $  ; wavelength spread
  ,SampleDim=SampleDim      $  ; Sample dimensions [Length,Width,thickness] where Length is parallel to the beam in mm
  ,thetaS=thetaS      $        ; the specular angle of the beam on the sample (for detector bank0) in degrees
  ,Lsa=Lsa                  $  ; Sample-HOPG distance in mm
  ,AnalyzerDim=AnalyzerDim  $  ; HOPG dimensions [width, Height, thickness] in mm
  ,DetectorDim=DetectorDim  $  ; Detector dimensions [width, Height, thickness] in mm
  ,thetaA=thetaA        $      ; HOPG theta in degrees
  ,Lad=Lad                  $  ; HOPG-Detector distance in mm
  ,nPoints=nPoints $
  ,noPlotFlag=noPlotFlag $
  ,_Extra=etc
  
  ; Set some default values if needed
  ; Default distances are for the first (upstream) analyzer/detector in Bank0!
  if (N_elements(wavelength) ne 1) then wavelength = 6.00              ; mean wavelength in angstroms  
  if (n_elements(SampleDim) ne 3) then SampleDim = [50.0,10.0,5.0]     ; sample [Length, Width, thickness]
  if (N_elements(thetaS) ne 1) then thetaS = 10.0                      ; specular angle for detector bank0
  if (N_elements(AnalyzerDim) ne 3) then AnalyzerDim = [12.0,30.0,1.0] ; HOPG [width, height, thickness]
  if (N_elements(DetectorDim) ne 3) then DetectorDim = [6.00,30.0,2.0] ; Detector [width, height, thickness]
  if (N_elements(Lsa) ne 1) then Lsa = 3572.81                         ; Sample to the HOPD in question
  if (N_elements(Lad) ne 1) then Lad = 18.100                          ; HOPD to its accompanying detector
  if (N_elements(thetaA) ne 1) then thetaA = 63.1                      ; specular angle for detector bank0
  if N_elements(nPoints) eq 0 then nPoints = 101
  
  thetaS = thetaS*!dtor                 ; to radians
  thetaA  = thetaA*!dtor
  SampleDim   *= 0.001                  ; mm to m
  AnalyzerDim *= 0.001
  DetectorDim *= 0.001
  Lsa *= 0.001
  Lad *= 0.001
  
  hOverMn = 3956.034      ; tof = d*wavelength/hOverMn

  ;######################################################################
  ; Based on the geometry of the instrument, determine (min, max) limits along the x, y and z axes
  ; for each of the components
  ; Then, generate random points that lie within the set limits.

  ; Assume a coordinate system where 
  ; y is along the scattered beam (after sample)
  ; z is into the paper (along sample width)
  ; x is pointing up along height of HOPG/Detector
  ; Hence the incident and scattered beam on the sample lie within the x-y plane

  ns = nPoints
  Ls = SampleDim[0] ; Sample is assume to have dimensions of Len,Width and thickness
  Ws = SampleDim[1]
  Ts = SampleDim[2]

  xMin = -Ts/2.0*Cos(thetaS) - Ls/2.0*Sin(thetaS)
  xMax =  Ts/2.0*Cos(thetaS) + Ls/2.0*Sin(thetaS)
  seed = !NULL
  SxCoords = Randomu(seed, ns)*(xMax-xMin) + xMin  ; generate ns random values between xMin and xMax

  yMin = -Ts/2.0*Sin(thetaS) - Ls/2.0*Cos(thetaS)
  yMax =  Ts/2.0*Sin(thetaS) + Ls/2.0*Cos(thetaS)
  seed = !NULL
  SyCoords = Randomu(seed, ns)*(yMax-yMin) + yMin  ; generate ns random values between yMin and yMax

  zMin = -Ws/2.0
  zMax =  Ws/2.0
  seed = !NULL
  SzCoords = Randomu(seed, ns)*(zMax-zMin) + zMin  ; generate ns random values between zMin and zMax

  ; Randomly generate points within a single HOPG/analyzer using the coordinates defined above
  ; HOPG has dimensions of Width(Wa), Height(Ha) and thickness(Ta)
  na = nPoints
  Wa = AnalyzerDim[0]
  Ha = AnalyzerDim[1]
  Ta = AnalyzerDim[2]

  xMin = -Ha/2.0
  xMax =  Ha/2.0
  seed = !NULL
  AxCoords = Randomu(seed, na)*(xMax-xMin) + xMin  ; generate na random values between xMin and xMax

  yMin = Lsa - Wa/2.0*Cos(thetaA)
  yMax = Lsa + Wa/2.0*Cos(thetaA)
  seed = !NULL
  AyCoords = Randomu(seed, na)*(yMax-yMin) + yMin  ; generate na random values between yMin and yMax

  zMin = -Wa/2.0*Sin(thetaA) - Ta/2.0*Cos(thetaA)
  zMax =  Wa/2.0*Sin(thetaA) + Ta/2.0*Cos(thetaA)
  seed = !NULL
  AzCoords = Randomu(seed, na)*(zMax-zMin) + zMin  ; generate na random values between zMin and zMax

  ; Randomly generate points within a single scintillator detector using the coordinates defined above
  ; Detector has dimensions of Width(Wa), Height(Ha) and thickness(Ta)
  nd = nPoints
  Wd = DetectorDim[0]
  Hd = DetectorDim[1]
  Td = DetectorDim[2]

  xMin = -Hd/2.0
  xMax =  Hd/2.0
  seed = !NULL
  DxCoords = Randomu(seed, nd)*(xMax-xMin) + xMin  ; generate nd random values between xMin and xMax

  yMin = Lsa + Lad*Cos(2*thetaA) - Wd/2.0
  yMax = Lsa + Lad*Cos(2*thetaA) + Wd/2.0
  seed = !NULL
  DyCoords = Randomu(seed, nd)*(yMax-yMin) + yMin  ; generate nd random values between yMin and yMax

  zMin =  Lad*Sin(2*thetaA) - Td/2.0
  zMax =  Lad*Sin(2*thetaA) + Td/2.0
  seed = !NULL
  DzCoords = Randomu(seed, nd)*(zMax-zMin) + zMin  ; generate nd random values between zMin and zMax

  ;######################################################################
  ; Now calculate the distance between the components using the points generated above
  ; to determine the total sample-dector distance.
  ; Accumulate the result to obtain a random distribution of sample-detector distances
  tic     ; start clock to estimate calculation time
  Lsd = []
  for i=0,ns-1 do begin
    for j=0,na-1 do begin
      for k=0,nd-1 do begin
        Li=Sqrt((AxCoords[j]-SxCoords[i])^2+(AyCoords[j]-SyCoords[i])^2+(AzCoords[j]-SzCoords[i])^2)+$
          Sqrt((DxCoords[k]-AxCoords[j])^2+(DyCoords[k]-AyCoords[j])^2+(DzCoords[k]-AzCoords[j])^2)
        Lsd = [Lsd,Li]
      endfor
    endfor
  endfor
  toc     ; end clock
  print, n_elements(Lsd)  ; print total number of distances evaluated

  ;######################################################################
  ; Use statistical analysis to determine mean and standard deviation of
  ; the distance and hence tof
  moments = Moment(Lsd)
  Print,''
  Print,'##################################################'
  Print,'*** CANDOR ***
  Print,'##################################################'
  fmt1 = '(F6.1)'
  fmt2 = '(F6.3)'
  sigmaLambda = moments[0]*DeltaLambda/hOverMn
  results='Wavelength = '+String(wavelength,format=fmt1)+' $\AA$'
  results = [results,'Mean S-D dist = '+String(moments[0]*1000,format=fmt1)+' mm']
  results = [results,'$\sigma$ of S-D dist = '+String(Sqrt(moments[1])*1000,format=fmt1)+' mm']
  results = [results,'Mean TOF      = '+String(moments[0]*wavelength/hOverMn*1000.0,format=fmt2)+' ms']
  results = [results,'$\sigma$ TOF         = '+String(Sqrt(moments[1])*wavelength/hOverMn*1000.0,format=fmt2)+' ms']
  results = [results,'$\Delta t/t$    = '+string(Sqrt(moments[1])/moments[0]*100, format='(F4.2)')+' %']
  results = [results,'$\sigma (due to \Delta\lambda)$  = '+String(sigmaLambda*1000.0,format=fmt2)+' ms']
  for i=0,6 do Print,results[i]
  Print,'##################################################'

  if (Keyword_set(noPlotFlag)) then Return
  ;######################################################################
  ; Histogram the results to enable a plot of the distribution of distances and times
  frequency = Histogram(Lsd, nbins=51, locations=distances)
  tof = distances*wavelength/hOverMn

  xtitle='L$_{samp-detector}$ (m)'
  ytitle="'Intensity'"
  title='CANDOR: S-D Distances for $\lambda$ = '+string(Wavelength,format='(F5.2)')+' $\AA$'
  p1 = Plot(distances,frequency,ytitle=ytitle,xtitle=xtitle,title=title, $
    dim=[1000,600],linestyle=' ',symbol='diamond',color=red)
  xtitle='TOF$_{samp-detector}$ (s)'
  title='CANDOR: S-D TOF for $\lambda$ = '+String(Wavelength,format='(F5.2)')+' $\AA$'
  p2 = Plot(tof,frequency,ytitle=ytitle,xtitle=xtitle,title=title, $
    dim=[1000,600],linestyle=' ',symbol='diamond',color=red)
  void = Text(0.5,0.7,results[[0,1,2]],/normal,target=p1,font_name='Courier')
  void = Text(0.5,0.7,results[[0,3,4,5,6]],/normal,target=p2,font_name='Courier')
end

;###############################################################
; Driver procedure for the TOF Estimate
; Load the CANDOR paramters (wavelength and HOPG/detector distances) from a text file
; Specify which detector index is of interest to evaluate
; and also the number of points (default is 101)
pro drive_TofEstForCANDOR, detector_index, nPoints=nPoints
if (n_params() ne 1) then detector_index = 0

; retrieve CANDOR parameters file
path = sourcepath()+path_sep()
CANDOR_parFile = 'CANDOR_lambda_distances_bank0.txt'
parfile = path+CANDOR_parFile
if (~File_test(parfile,/read)) then begin
  parfile = dialog_pickfile(dialog_parent=0L, title='Select CANDOR parameter (wavelength, distances) file' $
    ,/read,filter='*.txt',path=path)
  if (~File_test(parfile,/read)) then return  ;Parameter file is required
endif

; read CANDOR parameters from text file
nlines = File_lines(parfile)
buffer = fltarr(6,nlines)
Openr,lun,parfile,/get_lun
Readf, lun, buffer
Free_lun,lun,/force
;index = intarr(nlines)
lambda = fltarr(nlines)
DeltaLambda = fltarr(nlines)
Lsa = fltarr(nlines)
lad = fltarr(nlines) 
Lsd = fltarr(nlines)
for i=0,nlines-1 do begin
  index = fix(buffer[0,i])
  lambda[index]       = buffer[1,i]
  DeltaLambda[index] = buffer[2,i]
  Lsa[index]          = buffer[3,i]
  lad[index]          = buffer[4,i]
  Lsd[index]          = buffer[5,i]
endfor

; which detector index are we interested in evaluating?
detector_index = 0 > detector_index  ; must be >= 0
detector_index = detector_index < 53 ; must be <= 53

Tofestforcandor, lambda[detector_index], DeltaLambda=DeltaLambda[detector_index] $
               , Lsa=Lsa[detector_index], Lad=Lad[detector_index], nPoints=nPoints
end
