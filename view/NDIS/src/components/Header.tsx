import styles from './Header.module.css';
import { Link } from 'react-router-dom';

function Header() {
    return (
        <div className={styles.header}>
            <p>VHUNTER</p>
            <div className={styles.links}>
                <Link to={""}>TECHNOLOGY</Link>
                <Link to={""}>ABOUT US </Link>
                <Link to={"/home"}>HOME</Link>
            </div>
        </div>
    )
}

export default Header