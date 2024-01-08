<template>
    <div class="head1">
        <h1 class="inline-block text-black text-xl">Accounts</h1>
        <button class="inline-block text-white font-bold text-xs bg-blue-600 p-2 rounded-2xl" @click="addAccount">+ ACCOUNT</button>
    </div>
    <div v-for="account in allAccounts" :key="account" class="container mt-8 m-4 p-1">
        <div class="flex space-between">
            <h1 class="flex-1 font-bold">{{ account[0] }}</h1>
            <h1 class="flex-1 font-bold" @click="clickEdit(account)">:</h1>
        </div>
        <div class="inner-content">
            <div class="left-content">
                <h5>{{ account[4] }}</h5>
                <h5>{{ account[3] }}</h5>
            </div>
            <div class="right-content">
                <button>:</button>
            </div>
        </div>
    </div>

</template>
<script setup lang="ts">
    import { ref, defineEmits, onMounted, computed } from 'vue';
    import axios from 'axios';

    const user_id = '6593ccdf025b256e0ffe24e8';
    const allAccounts = ref([]);
    let editData = ref(null)

    const fetchAccount = async () => {
    try {
        const response = await axios.get(`http://localhost:8080/api/v1/${user_id}`);
        allAccounts.value = response.data.list_account_name;
        console.log('Server response:', response.data.list_account_name
);
    } catch (error) {
        console.error('Error fetching account information:', error);
    }
    };
    const clickEdit = (itemEdit: any) => {
            editData = itemEdit,
            emits("edit-account", editData);
            console.log("edit-account:",editData)
    }

    onMounted(() => {
    fetchAccount();
    });

    const emits = defineEmits(['add-account', 'edit-account']);

    const addAccount = () => {
    emits('add-account');
    };
</script>
<style scoped>
    .head1{
        padding-top: 20px;
        padding-left: 30px;
        display: flex;
        justify-content: space-between;
        width: 95%;
    }
    .container{
    background-color: rgb(193, 184, 236);
    border-radius: 20px;
    }

    .inner-content {
    display: flex;
    background-color: gainsboro;
    justify-content: space-between;
    margin: 20px; /* Adjust the margin as needed */
    }

    .left-content {
    margin-right: 20px; /* Adjust the margin as needed */
    }

    .right-content {
    display: flex;
    padding-right: 20px;
    }
</style>