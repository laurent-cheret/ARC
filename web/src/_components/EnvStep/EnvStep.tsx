import { StepOutput, TaskGrid } from '../../types/types';
import Grid from '../Grid/Grid';
import styles from './styles.module.scss';

export default function EnvStep({ stepOutput }: { stepOutput: StepOutput }) {
  return (
    <div className={styles.envOutput}>
      <h3>
        step: {stepOutput.step} action: {stepOutput.action_name}
      </h3>

      <div className="examples-container">
        {stepOutput.current_grids.map((gridList: TaskGrid[], index: number) => (
          <div className="grid-column" key={index}>
            {gridList.map((grid: TaskGrid, index2: number) => (
              <Grid taskGrid={grid} key={index2}></Grid>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
