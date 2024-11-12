import { forwardRef } from 'react';
import { StepOutput, TaskGrid } from '../../types/types';
import Grid from '../Grid/Grid';
import styles from './styles.module.scss';

const EnvStep = forwardRef<HTMLDivElement, { stepOutput: StepOutput }>(({ stepOutput }, ref) => {
  return (
    <div className={styles.envOutput} ref={ref}>
      <h5 className="title headline-small on-surface-variant">
        Step {stepOutput.step}, action: {stepOutput.action_name}
      </h5>

      <div className="examples-container hide-scroll">
        {stepOutput.current_grids.map((gridList: TaskGrid[], index: number) => (
          <div className="grid-column" key={index}>
            {gridList.map((grid: TaskGrid, index2: number) => (
              <div className="grid-container">
                <Grid taskGrid={grid} key={index2}></Grid>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
});

export default EnvStep;
