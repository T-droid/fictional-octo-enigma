import { useState } from "react";
import UploadService from "../services/FileUploadService";


interface Prediction {
  row: number;
  prediction: string;
}

interface AttackCount {
    type: string;
    count: number;
}

const FileUpload: React.FC = () => {
    const [currentFile, setCurrentFile] = useState<File>();
    const [progress, setProgress] = useState<number>(0);
    const [message, setMessage] = useState<string>("");
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [isLive, setIsLIve] = useState(false);
    const [ws, setWs] = useState<WebSocket | null>(null)

    const selectFile = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { files } = event.target;
        const selectedFiles = files as FileList;
        setCurrentFile(selectedFiles?.[0]);
        setProgress(0);
        setPredictions([]);
    };

    const upload = () => {
        setProgress(0);
        if (!currentFile) return;
    
        UploadService.upload(currentFile, (event: any) => {
          setProgress(Math.round((100 * event.loaded) / event.total));
        })
          .then((response) => {
            setPredictions(response.data.predictions);
            setMessage("File processed successfully!");
          })
          .catch((err) => {
            setProgress(0);
            if (err.response && err.response.data && err.response.data.detail) {
                setMessage(err.response.data.detail);
            } else {
                setMessage("Could not process the file!");
                console.log(err);
            }
            setCurrentFile(undefined);
          });
    };
    const getAttackCounts = (predictions: Prediction[]): AttackCount[] => {
        const counts = predictions.reduce((acc: {[key: string]: number}, curr) => {
            acc[curr.prediction] = (acc[curr.prediction] || 0) + 1;
            return acc;
        }, {});

        return Object.entries(counts).map(([type, count]) => ({
            type,
            count
        }));
    };
    
    const startLiveMonitoring = () => {
        setIsLIve(true);
        const websocket = new WebSocket('ws://localhost:8000/ws');

        websocket.onmessage = (event) => {
            const prediction = JSON.parse(event.data);
            setPredictions(prev => [...prev, {
                row: prev.length,
                prediction: prediction.attack_type
            }]);
        }

        setWs(websocket)
    };

    const stopLiveMonitoring = () => {
        setIsLIve(false);
        if (ws) {
            ws.close()
            setWs(null)
        }
    };

    return (
        <div className="container">
            <div className="row mb-3">
                <div className="col-8">
                    <label className="btn btn-default p-0">
                        <input type="file" accept=".csv" onChange={selectFile} />
                    </label>
                </div>
                <div className="col-4">
                    <button
                        className="btn btn-success btn-sm"
                        disabled={!currentFile}
                        onClick={upload}
                    >
                        Upload and Analyze
                    </button>
                </div>
            </div>
            <div className="row mb-3">
                <div className="col-8">
                    <label className="btn btn-default p-0">
                        <input type="file" accept=".csv" onChange={selectFile} />
                    </label>
                </div>
                <div className="col-4">
                    <button 
                        className={`btn ${isLive ? 'btn-danger' : 'btn-success'}`}
                        onClick={isLive ? stopLiveMonitoring : startLiveMonitoring}
                    >
                        {isLive ? 'Stop Live Monitoring' : 'Start Live Monitoring'}
                    </button>
                </div>
            </div>

            {currentFile && (
                <div className="progress mb-3">
                    <div
                        className="progress-bar progress-bar-info"
                        role="progressbar"
                        aria-valuenow={progress}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        style={{ width: progress + "%" }}
                    >
                        {progress}%
                    </div>
                </div>
            )}

            {message && (
                <div className="alert alert-secondary mb-3" role="alert">
                    {message}
                </div>
            )}

{predictions.length > 0 && (
                <div className="card">
                    <div className="card-header">Attack Type Distribution</div>
                    <div className="card-body">
                        <div className="table-responsive">
                            <table className="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Attack Type</th>
                                        <th>Occurrences</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {getAttackCounts(predictions).map((attack) => (
                                        <tr key={attack.type}>
                                            <td>{attack.type}</td>
                                            <td>{attack.count}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default FileUpload;
