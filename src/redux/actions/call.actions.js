import * as types from "../constants/call.constants";
import api from "../api";

const getSingleCall = (callId) => async (dispatch) => {
  dispatch({ type: types.GET_SINGLE_CALL_REQUEST, payload: null });
  try {
    const res = await api.get(`calls/${callId}`);
    console.log("RESSSPONSE", res);
    dispatch({
      type: types.GET_SINGLE_CALL_REQUEST_SUCCESS,
      payload: res.data,
    });
  } catch (error) {
    dispatch({ type: types.GET_SINGLE_CALL_REQUEST_FAILURE, payload: error });
  }
};

export const callActions = {
  getSingleCall,
};
