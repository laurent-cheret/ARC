import { forwardRef } from 'react';
import { StepOutput, TaskGrid } from '../../types/types';
import Grid from '../Grid/Grid';
import styles from './styles.module.scss';
import { Chip } from '@mui/material';

const EnvStep = forwardRef<HTMLDivElement, { stepOutput: StepOutput }>(({ stepOutput }, ref) => {
  return (
    <div className={styles.envOutput} ref={ref}>
      <h5 className="title headline-small on-surface-variant">
        Step {stepOutput.step}, action:&nbsp;&nbsp;
        <Chip
          label={stepOutput.action_name}
          sx={{
            fontWeight: 400,
            bgcolor: 'var(--secondaryContainer)',
            color: 'var(--onSecondaryContainer)',
            '& .MuiChip-deleteIcon': {
              color: 'var(--onSecondaryContainer)',
            },
          }}
        />
      </h5>

      <div className="examples-container hide-scroll">
        {stepOutput.current_grids.map((gridList: TaskGrid[], index: number) => (
          <div key={index} className="grid-column">
            {gridList.map((grid: TaskGrid, index2: number) => (
              <div key={index2} className="grid-container">
                <Grid taskGrid={grid}></Grid>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
});

export default EnvStep;
