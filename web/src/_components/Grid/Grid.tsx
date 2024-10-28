import { TaskGrid } from '../../types/types';
import styles from './styles.module.scss';

export default function Grid({ taskGrid }: { taskGrid: TaskGrid }) {
  return (
    <div className={styles.taskGrid}>
      {taskGrid.map((row, indexRow) => (
        <div key={indexRow} className="row">
          {row.map((cell, indexCell) => (
            <div key={indexCell} className={['cell', 'c' + cell].join(' ')}></div>
          ))}
        </div>
      ))}
    </div>
  );
}
