import Chatbot from "./Chatbot";
import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <Chatbot /> {/* Fügen Sie die Chatbot-Komponente hier ein */}
      </header>
    </div>
  );
}

export default App;

