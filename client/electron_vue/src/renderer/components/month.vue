<template>
    <p class="month" @click="toggleMonthPicker">{{ date.month }} {{ date.year }}</p>
    <month-picker-input v-if="showMonthPicker" class="form" :no-default="true" @change="showDate"></month-picker-input>
  </template>
  
  <script>
  import { MonthPickerInput } from 'vue-month-picker'
  
  export default {
    data() {
    return {
     date: {
      from: null,
      to: null,
      month: 1,
      year: 2024
     },
     showMonthPicker: false,
    };
   },
   components: {
    MonthPickerInput
   },
   methods: {
    showDate (date) {
     this.date = date
     this.$emit('send-month', this.date.month);
     this.$emit('send-year', this.date.year);
     this.showMonthPicker = false;
    },
    toggleMonthPicker() {
      this.showMonthPicker = !this.showMonthPicker;
    },
    },
  }
  </script>
<style scoped>
.form{
    position: relative;
    z-index: 999;
}
.month{
  color: white;
  cursor: pointer;
}
</style>