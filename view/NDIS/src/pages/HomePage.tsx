import FileUpload from "../components/FileUpload";
import Header from "../components/Header";

function HomePage() {
  return (
    <div style={{backgroundColor: 'black'}}>
    <div className="container" style={{ width: "600px" }}>
      <Header />
      <div className="my-3">
        <h3>Welcome To NDIS</h3>
        <div className="alert alert-info mt-3">
          <h5>About This Application</h5>
          <p>
            This application helps you process NDIS (National Disability Insurance Scheme) files.
            Simply upload your NDIS documents, and we'll help you manage and organize them efficiently.
          </p>
          <ul className="mb-0">
            <li>Upload NDIS related documents</li>
            <li>Process and validate files</li>
            <li>Get organized results instantly</li>
          </ul>
        </div>
      </div>

      <FileUpload />
    </div>
    </div>
  );
}

export default HomePage;