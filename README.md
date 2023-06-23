# Mel-Frequency Cepstrum Coefficients (MFFC)

## Table of contents
1. [Introduction](#p1)
2. [Mel-Frequency Cepstrum Coefficients Algorithm](#p2)
3. [MFFC results](#p3)
4. [Cats and Dogs audio classification](#p4)

## Introduction <a name="p1" /></a>

<p align="justify">
Mel-Frequency Cepstral Coefficients (MFCC) comes from combines two analysis:  <b>Cepstral and Mel-Frequency Analysis. </b><br/><br/>
	&emsp; &emsp;1. <b>Cepstral Analysis</b> aims to extract envelop of signal which carries the most relevant  information. It uses IFT (Inverse Fourier Transformation) and LPF (Low Pass Filter) to extract coefficient representing signal envelop of Log Power Signal Spectrum.<br/><br/>
&emsp;&emsp;<img src="https://github.com/FilipTirnanic96/mffc_extraction/assets/24530942/450e9e03-df93-4e41-9d1b-54b42544a72b" height="300" width="450">
 <br/>
 &emsp; &emsp;2. <b>Mel-Frequency Analysis</b> treats signal as human auditory system. It passes Signal spectrum though Mel filter which estimates human auditory system filtering. Example of Mel filter is presented in the picture below.<br/> <br/>
<img src="https://github.com/FilipTirnanic96/mffc_extraction/assets/24530942/8ade4eb5-3406-4d58-bfa2-f4daa088447a" height="300" width="450">
<br/>
 </p> 

## Mel-Frequency Cepstrum Coefficients Algorithm <a name="p2" /></a>
<p align="justify">
MFCC algorithm is composed of 8 steps:<br/><br/>
&emsp; &emsp;1. <b>Framing</b> - Signal is framed into chunks often with length of 20-40ms <br/>
&emsp; &emsp;2. <b>Windowing</b> - Frames are windowed with Hamming window <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mffc_extraction/assets/24530942/f2f0f57f-dd08-4f4a-9d5e-d189e6be9758" height="60" width="300"><br/>
&emsp; &emsp;3. Calculating <b>Power Spectrum</b> of each frame <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mffc_extraction/assets/24530942/510790d5-4301-405f-bcc9-a7a5959a19c7" height="70" width="300"><br/>
&emsp; &emsp;4. Filtering each Power Spectrum with <b>Mel filter</b> <br/>
&emsp; &emsp;5. Covert Mel Spectrum from <b>amplitude to dB</b> <br/>
&emsp; &emsp;6. Using <b>DCT2</b> transformation on each Mel Spectrum<br/>  
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mffc_extraction/assets/24530942/7665f08f-1717-45b3-a2b4-d02447883e9d" height="90" width="450"><br/>
&emsp; &emsp;7. <b>Liftering of MFCC</b> using sin window <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mffc_extraction/assets/24530942/406b274b-4d6d-49dd-8ccd-c82b278aebdc" height="70" width="220"><br/>  
&emsp; &emsp;8. Taking <b>2. - 13 coefficient</b><br/><br/>

Details of implementation can be found in <b><i>./core_functions/</i></b> folder. <br/>
</p>

## MFFC results <a name="p2" /></a>
<p align="justify">
  
</p>
