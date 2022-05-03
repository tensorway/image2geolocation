import React, { useState } from 'react';
import './FileUpload.css'
export const FileUploadPage = () => {

    const [selectedFile1, setSelectedFile1] = useState('');
    const [isFile1Picked, setIsFile1Picked] = useState(false);
    const [selectedFile2, setSelectedFile2] = useState('');
    const [isFile2Picked, setIsFile2Picked] = useState(false);
    const [selectedFile3, setSelectedFile3] = useState('');
    const [isFile3Picked, setIsFile3Picked] = useState(false);
    const [selectedFile4, setSelectedFile4] = useState('');
    const [isFile4Picked, setIsFile4Picked] = useState(false);

    const [retImage, setRetImage] = useState("")

    const [pickedImage1, setPickedImage1] = useState("")
    const [pickedImage2, setPickedImage2] = useState("")
    const [pickedImage3, setPickedImage3] = useState("")
    const [pickedImage4, setPickedImage4] = useState("")

    const [errMsg, setErrMsg] = useState("")

    const restartImage = (event: any) => {
        setSelectedFile1('')
        setSelectedFile2('')
        setSelectedFile3('')
        setSelectedFile4('')
        setIsFile1Picked(false)
        setIsFile2Picked(false)
        setIsFile3Picked(false)
        setIsFile4Picked(false)
        setRetImage("")
    }

    const changeHandler1 = (event: any) => {
        if (event.target.files && event.target.files[0]) {
            setPickedImage1(URL.createObjectURL(event.target.files[0]));
            setSelectedFile1(event.target.files[0]);
            setIsFile1Picked(true);
        }
    };

    const changeHandler3 = (event: any) => {
        if (event.target.files && event.target.files[0]) {
            setPickedImage3(URL.createObjectURL(event.target.files[0]));
            setSelectedFile3(event.target.files[0]);
            setIsFile3Picked(true);
        }
    };

    const changeHandler4 = (event: any) => {
        if (event.target.files && event.target.files[0]) {
            setPickedImage4(URL.createObjectURL(event.target.files[0]));
            setSelectedFile4(event.target.files[0]);
            setIsFile4Picked(true);
        }
    };

    const changeHandler2 = (event: any) => {
        if (event.target.files && event.target.files[0]) {
            setPickedImage2(URL.createObjectURL(event.target.files[0]));
            setSelectedFile2(event.target.files[0]);
            setIsFile2Picked(true);
        }
    };

    const handleSubmission = () => {

        const formData = new FormData();

        if (!isFile1Picked && !isFile2Picked && !isFile3Picked && !isFile4Picked) {
            setErrMsg("Please choose at least one image.")
            return
        } else {
            setErrMsg("")
        }

        if (isFile1Picked) {
            formData.append('files', selectedFile1);
        }
        if (isFile2Picked) {
            formData.append('files', selectedFile2);
        }
        if (isFile3Picked) {
            formData.append('files', selectedFile3);
        }
        if (isFile4Picked) {
            formData.append('files', selectedFile4);
        }

        setRetImage("")
        fetch(
            'http://localhost:8000/uploader/upload_images/',
            {
                method: 'POST',
                body: formData,
            }

        )

            .then((response) => response.json())
            .then((result) => {
                setRetImage(result['img'])
            })

            .catch((error) => {
                console.error('Error:', error);
            });
    };
    return (
        retImage === "" ?
            <div className='root-cont-images'>
                <p className='error-msg'>{errMsg}</p>
                <div className='images-loaders-container'>
                    <div>
                        {isFile1Picked ? (
                            <div className='img-details-container'>
                                <img className='loaded-img' src={pickedImage1} alt="" />
                            </div>
                        ) : (
                            <div className='placeholder-image'>
                                <label htmlFor="pc1" className='modal-label'>Load image</label>
                                <input type="file" id="pc1" className='input-hidden' onChange={changeHandler1} />
                            </div>
                        )}
                    </div>
                    <div>
                        {isFile2Picked ? (
                            <div className='img-details-container'>
                                <img className='loaded-img' src={pickedImage2} alt="" />
                            </div>

                        ) : (
                            <div className='placeholder-image'>
                                <label htmlFor="pc2" className='modal-label'>Load image</label>
                                <input type="file" id="pc2" className='input-hidden' onChange={changeHandler2} />
                            </div>
                        )}
                    </div>
                    <div>
                        {isFile3Picked ? (
                            <div className='img-details-container'>
                                <img className='loaded-img' src={pickedImage3} alt="" />
                            </div>

                        ) : (
                            <div className='placeholder-image'>
                                <label htmlFor="pc3" className='modal-label'>Load image</label>
                                <input type="file" id="pc3" className='input-hidden' onChange={changeHandler3} />
                            </div>
                        )}
                    </div>
                    <div>
                        {isFile4Picked ? (
                            <div className='img-details-container'>
                                <img className='loaded-img' src={pickedImage4} alt="" />
                            </div>

                        ) : (
                            <div className='placeholder-image'>
                                <label htmlFor="pc4" className='modal-label'>Load image</label>
                                <input type="file" id="pc4" className='input-hidden' onChange={changeHandler4} />
                            </div>
                        )}
                    </div>
                </div>
                <div>
                    <button onClick={handleSubmission} className="modal-label">Guess location</button>
                </div>
            </div> :
            <div className='global-res-cnt'><div className='show-results-container'>
                <img src={`data:image/jpg;base64,${retImage}`} className="ret-karta" alt="" />
                <button onClick={restartImage} className="modal-label" id='ret-karta-button'>Try with another location</button>
            </div></div>

    )
};
