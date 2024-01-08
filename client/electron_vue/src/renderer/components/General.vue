<template>
    <div class="store">
      <div class="profile1">
        <img src="../assets/img2.png" alt="">
        <h1 class="pt-4 pb-4 text-l font-bold">{{ profile.username }}</h1>
        <button @click="editProfile([profile.password, profile.number, profile.address])">Edit</button>
      </div>
      <div class="profile2">
        <div class="info name">
          <h2>Name</h2>
          <p>{{ profile.full_name }}</p>
        </div>
        <div class="info number">
          <h2>Phone num</h2>
          <p>{{ profile.number }}</p>
        </div>
        <div class="info dc">
          <h2>Address</h2>
          <p>{{ profile.address }}</p>
        </div>
        <div class="info email">
          <h2>Email</h2>
          <p>{{ profile.email }}</p>
        </div>
      </div>
    </div>
</template>
  
<script setup lang="ts">
import { ref, onMounted, defineEmits } from 'vue';
import axios from 'axios';

const user_id = '6593ccdf025b256e0ffe24e8';
const profile = ref({
    _id: '',
    password: '',
  full_name: '',
  username:'',
  name: '',
  number: '',
  address: '',
  email: '',
});
let editData = ref(null)

const fetchProfile = async () => {
  try {
    const response = await axios.get(`http://localhost:8080/api/v1/profile/${user_id}`);
    profile.value = response.data;
    console.log('Server:', profile.value);
  } catch (error) {
    console.error('Error fetching account information:', error);
  }
};

const editProfile = (itemEdit: any) => {
            editData = itemEdit,
            emits("edit-info", editData);
            console.log("edit-info:",editData)
        }

const emits = defineEmits(['edit-info']);

onMounted(() => {
  fetchProfile();
});
</script>

<style scoped>
    .store{
            margin: auto;
            padding: 20px;
            display: flex;
            align-items: center;
            text-align: center;

}

    .profile1 {
            width: 30%;
            height: 245px;
            align-items: center;
            justify-content: center;
        }

    .profile1 img {
            width: 100px;
            height: 100px;
        }

    .profile2 {
            width: 70%;
            padding: 0 20px;
        }

    h2 {
            font-size: 20px;
            margin-bottom: 5px;
            color: #333;
        }

    p {
            font-size: 16px;
            color: #555;
            margin-bottom: 10px;
        }

    .info {
            width: 100%;
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 5px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: space-between;
            height: 50px;
        }
    button{
            background-color: #4caf50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
            cursor: pointer;
            width: 70%;
            font-size: 16px;
        }
    img{
        border-radius: 50px;
    }
</style>