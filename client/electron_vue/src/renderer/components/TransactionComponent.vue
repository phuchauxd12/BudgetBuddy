<script setup lang="ts">
import TransactionBill from './TransactionBill.vue';
import TransactionHistory from './TransactionHistory.vue';
</script>
<template>

<section class="display">

<div class="bill pt-8 px-5">
    <TransactionBill/>
   
</div>
<div class="w-full">
    <TransactionHistory/>
</div>
</section>
</template>