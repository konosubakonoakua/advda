import axios from 'axios';
//创建axios的一个实例 
const axios_service = axios.create({
    baseURL:'http://localhost:23333/',//接口统一域名
    timeout: 6000*5 //设置超时
});
export { axiosService };