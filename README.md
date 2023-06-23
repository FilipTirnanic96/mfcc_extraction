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
&emsp;&emsp;<img src="https://github.com/FilipTirnanic96/mffc_extraction/assets/24530942/450e9e03-df93-4e41-9d1b-54b42544a72b" height="260" width="410">
 <br/>
 &emsp; &emsp;2. <b>Mel-Frequency Analysis</b> treats signal as human auditory system. It passes Signal spectrum though Mel filter which estimates human auditory system filtering. Example of Mel filter is presented in the picture below.<br/> <br/>
&emsp;&emsp;<img src="https://github.com/FilipTirnanic96/mffc_extraction/assets/24530942/8ade4eb5-3406-4d58-bfa2-f4daa088447a" height="250" width="400">
<br/>
 </p> 

## Mel-Frequency Cepstrum Coefficients Algorithm <a name="p2" /></a>
<p align="justify">
<ins>MFCC algorithm is composed of 8 steps:</ins><br/><br/>
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
&emsp; &emsp;8. Taking <b>2. - 13 coefficient</b><br/>

Details of implementation can be found in scipts <b><i>./core_functions/mffc_feature_extraction.py</i></b> and <b><i>./core_functions/mffc_utility_functions.py</i></b>. <br/>
</p>

## MFFC results <a name="p3" /></a>
<p align="justify">
Algorithm results are compared to mfcc extraction algorithm from <b>librosa package</b>. One representative signal (cat audio sample) is used as input signal and Spectrograms and MFFCs are generated using both algorithms (implemented one and librosa). Results, generated using script <b>./main_scrips/compare_impl_vs_librosa_mffc.py</b>, are shown below. <br/><br/>
&emsp; &emsp;- 	<ins><b> Input signal </b></ins> <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/ed0945e6-2312-41e0-a8a9-27359f5e34a3" height="200" width="350"><br/>

&emsp; &emsp;- 	<ins><b> Spectrograms </b></ins> <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/dd6a201f-201f-48a3-b405-090c436b5e0e" height="280" width="380">
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/6f3f4f4d-43b3-4e08-b1b0-ba81454450e9" height="280" width="380"><br/>		

&emsp; &emsp;- 	<ins><b> MFCCs </b></ins> <br/>
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/c77b0955-153d-44ca-9059-2c510a2387f5" height="300" width="400">
&emsp; &emsp;<img src="https://github.com/FilipTirnanic96/mfcc_extraction/assets/24530942/0ab541a6-947c-4570-822f-a2da8fce0262" height="300" width="400"><br/>		

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
