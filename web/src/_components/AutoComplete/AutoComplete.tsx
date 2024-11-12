import { Autocomplete, TextField } from '@mui/material';

export default function AutoCompleteChipList({
  primitives,
  demoActions,
  setDemoActions,
}: {
  primitives: string[];
  demoActions: string[];
  setDemoActions: Function;
}) {
  return (
    <>
      <Autocomplete
        multiple
        id="tags-outlined"
        options={primitives}
        getOptionLabel={(primitive) => primitive}
        value={demoActions}
        onChange={(_, value) => {
          setDemoActions(value);
        }}
        autoHighlight
        fullWidth
        size="small"
        isOptionEqualToValue={() => false}
        renderInput={(params) => <TextField {...params} label="DSL action list" placeholder="DSL actions list" />}
      />
    </>
  );
}
