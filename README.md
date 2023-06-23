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
&emsp;&emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/aa09d42a-79bf-4e7f-8caf-0bd6bc532216" height="260" width="410">

 <br/>
 &emsp; &emsp;2. <b>Mel-Frequency Analysis</b> treats signal as human auditory system. It passes Signal spectrum though Mel filter which estimates human auditory system filtering. Example of Mel filter is presented in the picture below.<br/> <br/>
&emsp;&emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/f4cec7b9-5b3e-4cd2-90a1-e7e1ee04ee33" height="250" width="400">
<br/>
 </p> 

## Mel-Frequency Cepstrum Coefficients Algorithm <a name="p2" /></a>
<p align="justify">
<ins>MFCC algorithm is composed of 8 steps:</ins><br/><br/>
&emsp; &emsp;1. <b>Framing</b> - Signal is framed into chunks often with length of 20-40ms <br/>
&emsp; &emsp;2. <b>Windowing</b> - Frames are windowed with Hamming window <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/384e8d45-8517-4112-8bb8-4bf78b86f636" height="60" width="300"><br/>
&emsp; &emsp;3. Calculating <b>Power Spectrum</b> of each frame <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/797fdce5-68c3-47a1-9413-2366133b82bf" height="70" width="300"><br/>
&emsp; &emsp;4. Filtering each Power Spectrum with <b>Mel filter</b> <br/>
&emsp; &emsp;5. Covert Mel Spectrum from <b>amplitude to dB</b> <br/>
&emsp; &emsp;6. Using <b>DCT2</b> transformation on each Mel Spectrum<br/>  
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/38a4282a-83a5-493b-bcb6-e68445835b99" height="90" width="450"><br/>
&emsp; &emsp;7. <b>Liftering of MFCC</b> using sin window <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/b5ccf1b8-d723-4bcf-b486-163519ff3d18" height="70" width="220"><br/>  
&emsp; &emsp;8. Taking <b>2. - 13 coefficient</b><br/>

Details of implementation can be found in scipts <b><i>./core_functions/mffc_feature_extraction.py</i></b> and <b><i>./core_functions/mffc_utility_functions.py</i></b>. <br/>
</p>

## MFFC results <a name="p3" /></a>
<p align="justify">
Algorithm results are compared to mfcc extraction algorithm from <b>librosa package</b>. One representative signal (cat audio sample) is used as input signal and Spectrograms and MFFCs are generated using both algorithms (implemented one and librosa). Results, generated using script <b>./main_scrips/compare_impl_vs_librosa_mffc.py</b>, are shown below. <br/><br/>
&emsp; &emsp;- 	<ins><b> Input signal </b></ins> <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/1b0d563d-c830-4c9d-bb0c-7e953b5cc353" height="200" width="350"><br/>

&emsp; &emsp;- 	<ins><b> Spectrograms </b></ins> <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/e2801845-2fdd-46df-b82b-181f5c7542d0" height="280" width="380">
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/a3e5106e-ac6a-4402-8b65-a0d8de662545" height="280" width="380"><br/>		

&emsp; &emsp;- 	<ins><b> MFCCs </b></ins> <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/eeab433a-d385-4026-af01-e14efeab5c40" height="300" width="400">
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/9479e312-6910-4a36-a126-ce69ee1bcffe" height="300" width="400"><br/>		

</p>

## Cats and Dogs audio classification <a name="p4" /></a>

<p align="justify">
The classification of Cats and Dogs audio dataset are preformed using implemented MFCCs extraction algorithm for feature extraction. Implementation can be found in script <b>./main_scrips/cats_and_dog_classification.py</b>. Data used for training and testing the classifier can be found in folder <b>./data/</b>. Classification is performed using next steps:<br/>
&emsp; &emsp;1. <b><i>Load train and test data</i></b> <br/>
&emsp; &emsp;2. <b><i>Pad data to same length</i></b> <br/>
&emsp; &emsp;3. <b><i>Extract MFFC from data</i></b> <br/>	
&emsp; &emsp;4. <b><i>Reduce dimensionality with PCA to 30 components</i></b> <br/>
&emsp; &emsp;5. <b><i>Train and test LogisticRegression classifier</i></b> <br/><br/>
Classifier reached 93% accuracy on train and test data. <ins>Confusion matrices are presented below:</ins><br/><br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/cd43fa8a-fd43-4377-8ee3-7458a601d57d" height="300" width="400">
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/257a88ac-7b1a-418e-85c2-ac4697e61f31" height="300" width="400"><br/>

</p>
