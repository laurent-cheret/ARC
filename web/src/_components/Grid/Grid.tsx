import { TaskGrid } from '../../types/types';
import styles from './styles.module.scss';

export default function Grid({ taskGrid }: { taskGrid: TaskGrid }) {
  const nbRows = taskGrid.length;
  const nbColumns = taskGrid[0].length;

  let gridStyle = 'horiz';
  if (nbRows > nbColumns) {
    gridStyle = 'vert';
  }

  return (
    <div className={styles.taskGrid}>
      {taskGrid.map((row, indexRow) => (
        <div key={indexRow} className={['row', gridStyle].join(' ')}>
          {row.map((cell, indexCell) => (
            <div key={indexCell} className={['cell', 'c' + cell].join(' ')}></div>
          ))}
        </div>
      ))}
    </div>
  );
}
