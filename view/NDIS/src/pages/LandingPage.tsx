import Header from "../components/Header";
import styles from "./LandingPage.module.css";
import dataBreach from "../assets/data-breach.webp";
import vulnerability from "../assets/vulnerability.jpg";
import services from "../assets/software-services.jpg";

function LandingPage() {
  return (
    <>
  <div>
    <Header />
        <div className={styles.container}>
            <div className={styles.text}>
                <h1>AI-DRIVEN</h1>
                <h1>CYBERSECURITY</h1>
                <h1>SOLUTIONS</h1>
                <p>In the cyber world, security is </p>
                <p>the most valuable and vulnerable thing a system has!</p>
                <p>SO what better than having tireless AI to take care of it</p>
                <div>
                    <button className={styles.explore_bt}>EXPLORE</button>
                    <button className={styles.contact_bt}>CONTACT US</button>
                </div>
            </div>
        </div>
    </div>
    <div className={styles.container2}>
        <div className={styles.card}>
            <img src={dataBreach} alt="data breaches" />
            <p>POTENTIAL DATA BREACHES</p>
        </div>
        <div className={styles.card}>
            <img src={vulnerability} alt="vulnerability" />
            <p>AVOID CODE VULNERABILITY</p>
        </div>
        <div className={styles.card}>
            <img src={services} alt="software services" />
            <p>+60 UNIQUE SERVICES FOR NEW LEVEL OF SECURITY</p>
        </div>
    </div>
  </>
  );
}

export default LandingPage;