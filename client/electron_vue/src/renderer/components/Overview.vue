

<template>
  <div class="grid">
    <div class="calender"></div>
    <div class="total_saving"></div>
    <div class="investment"></div>
    <div class="income"></div>
  </div>  
</template>
<style scoped>
.grid{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr 1fr 1fr;
  grid-template-rows: auto;
  grid-gap: 10px;
  margin: 20px;
}

.calender{ 
  border-radius: 15px;
  grid-column: 1/ span 3;
  grid-row: 1 / span 20;
  background-color: rgba(184, 240, 212, 0.54);
}
.total_saving{
  border-radius: 15px;
  grid-column: 4 / span 2;
  grid-row: 1 / span 10;
  background-color: rgba(184, 240, 212, 0.54);
}
.investment{
  border-radius: 15px;
  grid-column: 4 /span 2;
  grid-row: 11 / span 10;
  background-color: rgba(184, 240, 212, 0.54);
}
.income{
  border-radius: 15px;
  grid-column: 6/ span 3;
  grid-row: 1 / span 20;
  background-color: rgba(184, 240, 212, 0.54);
}
</style>