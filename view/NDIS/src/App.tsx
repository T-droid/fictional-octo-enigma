import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";
import AppRouter from "./routes/AppRouter";

const App: React.FC = () => {
  return (
    <div>
      <AppRouter />
    </div>
  );
}

export default App;
