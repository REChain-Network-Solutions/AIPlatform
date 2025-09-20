class O{constructor(t){this.onDataAvailable=t,this.audioContext=null,this.mediaStream=null,this.mediaStreamSource=null,this.workletNode=null}async start(t){try{this.audioContext=new AudioContext({sampleRate:24e3}),await this.audioContext.audioWorklet.addModule("/themes/default/assets/js/audio/audio-worklet-processor.js"),this.mediaStream=t,this.mediaStreamSource=this.audioContext.createMediaStreamSource(this.mediaStream),this.workletNode=new AudioWorkletNode(this.audioContext,"audio-worklet-processor"),this.workletNode.port.onmessage=a=>this.onDataAvailable(a.data.buffer),this.mediaStreamSource.connect(this.workletNode)}catch(a){console.error("Error in recorder start:",a),this.stop()}}stop(){this.mediaStream&&this.mediaStream.getTracks().forEach(t=>t.stop()),this.audioContext&&this.audioContext.state!=="closed"&&this.audioContext.close()}getMediaStreamSource(){return this.mediaStreamSource}}function k(){return k=Object.assign?Object.assign.bind():function(c){for(var t=1;t<arguments.length;t++){var a=arguments[t];for(var s in a)({}).hasOwnProperty.call(a,s)&&(c[s]=a[s])}return c},k.apply(null,arguments)}function E(c){const t=new Uint8Array(c);return window.btoa(String.fromCharCode(...t))}function T(c){const t=window.atob(c),a=t.length,s=new Uint8Array(a);for(let i=0;i<a;i++)s[i]=t.charCodeAt(i);return s.buffer}const b=new Map;function I(c,t){return async a=>{const s=b.get(c);if(s)return a.addModule(s);const i=new Blob([t],{type:"application/javascript"}),o=URL.createObjectURL(i);try{return await a.addModule(o),void b.set(c,o)}catch{URL.revokeObjectURL(o)}try{const n=`data:application/javascript;base64,${btoa(t)}`;await a.addModule(n),b.set(c,n)}catch{throw new Error(`Failed to load the ${c} worklet module. Make sure the browser supports AudioWorklets.`)}}}const R=I("raw-audio-processor",`
const BIAS = 0x84;
const CLIP = 32635;
const encodeTable = [
  0,0,1,1,2,2,2,2,3,3,3,3,3,3,3,3,
  4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
  5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
  6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
  6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
  6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
  6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7
];

function encodeSample(sample) {
  let sign;
  let exponent;
  let mantissa;
  let muLawSample;
  sign = (sample >> 8) & 0x80;
  if (sign !== 0) sample = -sample;
  sample = sample + BIAS;
  if (sample > CLIP) sample = CLIP;
  exponent = encodeTable[(sample>>7) & 0xFF];
  mantissa = (sample >> (exponent+3)) & 0x0F;
  muLawSample = ~(sign | (exponent << 4) | mantissa);
  
  return muLawSample;
}

class RawAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
              
    this.port.onmessage = ({ data }) => {
      switch (data.type) {
        case "setFormat":
          this.isMuted = false;
          this.buffer = []; // Initialize an empty buffer
          this.bufferSize = data.sampleRate / 4;
          this.format = data.format;

          if (globalThis.LibSampleRate && sampleRate !== data.sampleRate) {
            globalThis.LibSampleRate.create(1, sampleRate, data.sampleRate).then(resampler => {
              this.resampler = resampler;
            });
          }
          break;
        case "setMuted":
          this.isMuted = data.isMuted;
          break;
      }
    };
  }
  process(inputs) {
    if (!this.buffer) {
      return true;
    }
    
    const input = inputs[0]; // Get the first input node
    if (input.length > 0) {
      let channelData = input[0]; // Get the first channel's data

      // Resample the audio if necessary
      if (this.resampler) {
        channelData = this.resampler.full(channelData);
      }

      // Add channel data to the buffer
      this.buffer.push(...channelData);
      // Get max volume 
      let sum = 0.0;
      for (let i = 0; i < channelData.length; i++) {
        sum += channelData[i] * channelData[i];
      }
      const maxVolume = Math.sqrt(sum / channelData.length);
      // Check if buffer size has reached or exceeded the threshold
      if (this.buffer.length >= this.bufferSize) {
        const float32Array = this.isMuted 
          ? new Float32Array(this.buffer.length)
          : new Float32Array(this.buffer);

        let encodedArray = this.format === "ulaw"
          ? new Uint8Array(float32Array.length)
          : new Int16Array(float32Array.length);

        // Iterate through the Float32Array and convert each sample to PCM16
        for (let i = 0; i < float32Array.length; i++) {
          // Clamp the value to the range [-1, 1]
          let sample = Math.max(-1, Math.min(1, float32Array[i]));

          // Scale the sample to the range [-32768, 32767]
          let value = sample < 0 ? sample * 32768 : sample * 32767;
          if (this.format === "ulaw") {
            value = encodeSample(Math.round(value));
          }

          encodedArray[i] = value;
        }

        // Send the buffered data to the main script
        this.port.postMessage([encodedArray, maxVolume]);

        // Clear the buffer after sending
        this.buffer = [];
      }
    }
    return true; // Continue processing
  }
}
registerProcessor("raw-audio-processor", RawAudioProcessor);
`);function A(){return["iPad Simulator","iPhone Simulator","iPod Simulator","iPad","iPhone","iPod"].includes(navigator.platform)||navigator.userAgent.includes("Mac")&&"ontouchend"in document}class S{static async create({sampleRate:t,format:a,preferHeadphonesForIosDevices:s}){let i=null,o=null;try{const r={sampleRate:{ideal:t},echoCancellation:{ideal:!0},noiseSuppression:{ideal:!0}};if(A()&&s){const v=(await window.navigator.mediaDevices.enumerateDevices()).find(w=>w.kind==="audioinput"&&["airpod","headphone","earphone"].find(y=>w.label.toLowerCase().includes(y)));v&&(r.deviceId={ideal:v.deviceId})}const l=navigator.mediaDevices.getSupportedConstraints().sampleRate;i=new window.AudioContext(l?{sampleRate:t}:{});const h=i.createAnalyser();l||await i.audioWorklet.addModule("https://cdn.jsdelivr.net/npm/@alexanderolsen/libsamplerate-js@2.1.2/dist/libsamplerate.worklet.js"),await R(i.audioWorklet),o=await navigator.mediaDevices.getUserMedia({audio:r});const g=i.createMediaStreamSource(o),f=new AudioWorkletNode(i,"raw-audio-processor");return f.port.postMessage({type:"setFormat",format:a,sampleRate:t}),g.connect(h),h.connect(f),await i.resume(),new S(i,h,f,o)}catch(r){var n,e;throw(n=o)==null||n.getTracks().forEach(l=>l.stop()),(e=i)==null||e.close(),r}}constructor(t,a,s,i){this.context=void 0,this.analyser=void 0,this.worklet=void 0,this.inputStream=void 0,this.context=t,this.analyser=a,this.worklet=s,this.inputStream=i}async close(){this.inputStream.getTracks().forEach(t=>t.stop()),await this.context.close()}setMuted(t){this.worklet.port.postMessage({type:"setMuted",isMuted:t})}}const q=I("audio-concat-processor",`
const decodeTable = [0,132,396,924,1980,4092,8316,16764];

export function decodeSample(muLawSample) {
  let sign;
  let exponent;
  let mantissa;
  let sample;
  muLawSample = ~muLawSample;
  sign = (muLawSample & 0x80);
  exponent = (muLawSample >> 4) & 0x07;
  mantissa = muLawSample & 0x0F;
  sample = decodeTable[exponent] + (mantissa << (exponent+3));
  if (sign !== 0) sample = -sample;

  return sample;
}

class AudioConcatProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffers = []; // Initialize an empty buffer
    this.cursor = 0;
    this.currentBuffer = null;
    this.wasInterrupted = false;
    this.finished = false;
    
    this.port.onmessage = ({ data }) => {
      switch (data.type) {
        case "setFormat":
          this.format = data.format;
          break;
        case "buffer":
          this.wasInterrupted = false;
          this.buffers.push(
            this.format === "ulaw"
              ? new Uint8Array(data.buffer)
              : new Int16Array(data.buffer)
          );
          break;
        case "interrupt":
          this.wasInterrupted = true;
          break;
        case "clearInterrupted":
          if (this.wasInterrupted) {
            this.wasInterrupted = false;
            this.buffers = [];
            this.currentBuffer = null;
          }
      }
    };
  }
  process(_, outputs) {
    let finished = false;
    const output = outputs[0][0];
    for (let i = 0; i < output.length; i++) {
      if (!this.currentBuffer) {
        if (this.buffers.length === 0) {
          finished = true;
          break;
        }
        this.currentBuffer = this.buffers.shift();
        this.cursor = 0;
      }

      let value = this.currentBuffer[this.cursor];
      if (this.format === "ulaw") {
        value = decodeSample(value);
      }
      output[i] = value / 32768;
      this.cursor++;

      if (this.cursor >= this.currentBuffer.length) {
        this.currentBuffer = null;
      }
    }

    if (this.finished !== finished) {
      this.finished = finished;
      this.port.postMessage({ type: "process", finished });
    }

    return true; // Continue processing
  }
}

registerProcessor("audio-concat-processor", AudioConcatProcessor);
`);class M{static async create({sampleRate:t,format:a}){let s=null;try{s=new AudioContext({sampleRate:t});const o=s.createAnalyser(),n=s.createGain();n.connect(o),o.connect(s.destination),await q(s.audioWorklet);const e=new AudioWorkletNode(s,"audio-concat-processor");return e.port.postMessage({type:"setFormat",format:a}),e.connect(n),await s.resume(),new M(s,o,n,e)}catch(o){var i;throw(i=s)==null||i.close(),o}}constructor(t,a,s,i){this.context=void 0,this.analyser=void 0,this.gain=void 0,this.worklet=void 0,this.context=t,this.analyser=a,this.gain=s,this.worklet=i}async close(){await this.context.close()}}function F(c){return!!c.type}class C{static async create(t){let a=null;try{var s;const o=(s=t.origin)!=null?s:"wss://api.elevenlabs.io",n=t.signedUrl?t.signedUrl:o+"/v1/convai/conversation?agent_id="+t.agentId,e=["convai"];t.authorization&&e.push(`bearer.${t.authorization}`),a=new WebSocket(n,e);const r=await new Promise((w,y)=>{a.addEventListener("open",()=>{var p;const d={type:"conversation_initiation_client_data"};var m,u,_,x;t.overrides&&(d.conversation_config_override={agent:{prompt:(m=t.overrides.agent)==null?void 0:m.prompt,first_message:(u=t.overrides.agent)==null?void 0:u.firstMessage,language:(_=t.overrides.agent)==null?void 0:_.language},tts:{voice_id:(x=t.overrides.tts)==null?void 0:x.voiceId}}),t.customLlmExtraBody&&(d.custom_llm_extra_body=t.customLlmExtraBody),t.dynamicVariables&&(d.dynamic_variables=t.dynamicVariables),(p=a)==null||p.send(JSON.stringify(d))},{once:!0}),a.addEventListener("error",p=>{setTimeout(()=>y(p),0)}),a.addEventListener("close",y),a.addEventListener("message",p=>{const d=JSON.parse(p.data);F(d)&&(d.type==="conversation_initiation_metadata"?w(d.conversation_initiation_metadata_event):console.warn("First received message is not conversation metadata."))},{once:!0})}),{conversation_id:l,agent_output_audio_format:h,user_input_audio_format:g}=r,f=D(g??"pcm_16000"),v=D(h);return new C(a,l,f,v)}catch(o){var i;throw(i=a)==null||i.close(),o}}constructor(t,a,s,i){this.socket=void 0,this.conversationId=void 0,this.inputFormat=void 0,this.outputFormat=void 0,this.queue=[],this.disconnectionDetails=null,this.onDisconnectCallback=null,this.onMessageCallback=null,this.socket=t,this.conversationId=a,this.inputFormat=s,this.outputFormat=i,this.socket.addEventListener("error",o=>{setTimeout(()=>this.disconnect({reason:"error",message:"The connection was closed due to a socket error.",context:o}),0)}),this.socket.addEventListener("close",o=>{this.disconnect(o.code===1e3?{reason:"agent",context:o}:{reason:"error",message:o.reason||"The connection was closed by the server.",context:o})}),this.socket.addEventListener("message",o=>{try{const n=JSON.parse(o.data);if(!F(n))return;this.onMessageCallback?this.onMessageCallback(n):this.queue.push(n)}catch{}})}close(){this.socket.close()}sendMessage(t){this.socket.send(JSON.stringify(t))}onMessage(t){this.onMessageCallback=t,this.queue.forEach(t),this.queue=[]}onDisconnect(t){this.onDisconnectCallback=t,this.disconnectionDetails&&t(this.disconnectionDetails)}disconnect(t){var a;this.disconnectionDetails||(this.disconnectionDetails=t,(a=this.onDisconnectCallback)==null||a.call(this,t))}}function D(c){const[t,a]=c.split("_");if(!["pcm","ulaw"].includes(t))throw new Error(`Invalid format: ${c}`);const s=parseInt(a);if(isNaN(s))throw new Error(`Invalid sample rate: ${a}`);return{format:t,sampleRate:s}}const P={clientTools:{}},B={onConnect:()=>{},onDebug:()=>{},onDisconnect:()=>{},onError:()=>{},onMessage:()=>{},onAudio:()=>{},onModeChange:()=>{},onStatusChange:()=>{},onCanSendFeedbackChange:()=>{}};class L{static async startSession(t){var a;const s=k({},P,B,t);s.onStatusChange({status:"connecting"}),s.onCanSendFeedbackChange({canSendFeedback:!1});let i=null,o=null,n=null,e=null,r=null;if((a=t.useWakeLock)==null||a)try{r=await navigator.wakeLock.request("screen")}catch{}try{var l,h;e=await navigator.mediaDevices.getUserMedia({audio:!0});const m=(l=t.connectionDelay)!=null?l:{default:0,android:3e3};let u=m.default;var g;if(/android/i.test(navigator.userAgent))u=(g=m.android)!=null?g:u;else if(A()){var f;u=(f=m.ios)!=null?f:u}return u>0&&await new Promise(_=>setTimeout(_,u)),o=await C.create(t),[i,n]=await Promise.all([S.create(k({},o.inputFormat,{preferHeadphonesForIosDevices:t.preferHeadphonesForIosDevices})),M.create(o.outputFormat)]),(h=e)==null||h.getTracks().forEach(_=>_.stop()),e=null,new L(s,o,i,n,r)}catch(m){var v,w,y,p;s.onStatusChange({status:"disconnected"}),(v=e)==null||v.getTracks().forEach(u=>u.stop()),(w=o)==null||w.close(),await((y=i)==null?void 0:y.close()),await((p=n)==null?void 0:p.close());try{var d;await((d=r)==null?void 0:d.release()),r=null}catch{}throw m}}constructor(t,a,s,i,o){var n=this;this.options=void 0,this.connection=void 0,this.input=void 0,this.output=void 0,this.wakeLock=void 0,this.lastInterruptTimestamp=0,this.mode="listening",this.status="connecting",this.inputFrequencyData=void 0,this.outputFrequencyData=void 0,this.volume=1,this.currentEventId=1,this.lastFeedbackEventId=1,this.canSendFeedback=!1,this.endSession=()=>this.endSessionWithDetails({reason:"user"}),this.endSessionWithDetails=async function(e){if(n.status==="connected"||n.status==="connecting"){n.updateStatus("disconnecting");try{var r;await((r=n.wakeLock)==null?void 0:r.release()),n.wakeLock=null}catch{}n.connection.close(),await n.input.close(),await n.output.close(),n.updateStatus("disconnected"),n.options.onDisconnect(e)}},this.updateMode=e=>{e!==this.mode&&(this.mode=e,this.options.onModeChange({mode:e}))},this.updateStatus=e=>{e!==this.status&&(this.status=e,this.options.onStatusChange({status:e}))},this.updateCanSendFeedback=()=>{const e=this.currentEventId!==this.lastFeedbackEventId;this.canSendFeedback!==e&&(this.canSendFeedback=e,this.options.onCanSendFeedbackChange({canSendFeedback:e}))},this.onMessage=async function(e){switch(e.type){case"interruption":return e.interruption_event&&(n.lastInterruptTimestamp=e.interruption_event.event_id),void n.fadeOutAudio();case"agent_response":return void n.options.onMessage({source:"ai",message:e.agent_response_event.agent_response});case"user_transcript":return void n.options.onMessage({source:"user",message:e.user_transcription_event.user_transcript});case"internal_tentative_agent_response":return void n.options.onDebug({type:"tentative_agent_response",response:e.tentative_agent_response_internal_event.tentative_agent_response});case"client_tool_call":if(n.options.clientTools.hasOwnProperty(e.client_tool_call.tool_name))try{var r;const l=(r=await n.options.clientTools[e.client_tool_call.tool_name](e.client_tool_call.parameters))!=null?r:"Client tool execution successful.",h=typeof l=="object"?JSON.stringify(l):String(l);n.connection.sendMessage({type:"client_tool_result",tool_call_id:e.client_tool_call.tool_call_id,result:h,is_error:!1})}catch(l){n.onError("Client tool execution failed with following error: "+l?.message,{clientToolName:e.client_tool_call.tool_name}),n.connection.sendMessage({type:"client_tool_result",tool_call_id:e.client_tool_call.tool_call_id,result:"Client tool execution failed: "+l?.message,is_error:!0})}else{if(n.options.onUnhandledClientToolCall)return void n.options.onUnhandledClientToolCall(e.client_tool_call);n.onError(`Client tool with name ${e.client_tool_call.tool_name} is not defined on client`,{clientToolName:e.client_tool_call.tool_name}),n.connection.sendMessage({type:"client_tool_result",tool_call_id:e.client_tool_call.tool_call_id,result:`Client tool with name ${e.client_tool_call.tool_name} is not defined on client`,is_error:!0})}return;case"audio":return void(n.lastInterruptTimestamp<=e.audio_event.event_id&&(n.options.onAudio(e.audio_event.audio_base_64),n.addAudioBase64Chunk(e.audio_event.audio_base_64),n.currentEventId=e.audio_event.event_id,n.updateCanSendFeedback(),n.updateMode("speaking")));case"ping":return void n.connection.sendMessage({type:"pong",event_id:e.ping_event.event_id});default:return void n.options.onDebug(e)}},this.onInputWorkletMessage=e=>{this.status==="connected"&&this.connection.sendMessage({user_audio_chunk:E(e.data[0].buffer)})},this.onOutputWorkletMessage=({data:e})=>{e.type==="process"&&this.updateMode(e.finished?"listening":"speaking")},this.addAudioBase64Chunk=e=>{this.output.gain.gain.value=this.volume,this.output.worklet.port.postMessage({type:"clearInterrupted"}),this.output.worklet.port.postMessage({type:"buffer",buffer:T(e)})},this.fadeOutAudio=()=>{this.updateMode("listening"),this.output.worklet.port.postMessage({type:"interrupt"}),this.output.gain.gain.exponentialRampToValueAtTime(1e-4,this.output.context.currentTime+2),setTimeout(()=>{this.output.gain.gain.value=this.volume,this.output.worklet.port.postMessage({type:"clearInterrupted"})},2e3)},this.onError=(e,r)=>{console.error(e,r),this.options.onError(e,r)},this.calculateVolume=e=>{if(e.length===0)return 0;let r=0;for(let l=0;l<e.length;l++)r+=e[l]/255;return r/=e.length,r<0?0:r>1?1:r},this.getId=()=>this.connection.conversationId,this.isOpen=()=>this.status==="connected",this.setVolume=({volume:e})=>{this.volume=e},this.setMicMuted=e=>{this.input.setMuted(e)},this.getInputByteFrequencyData=()=>(this.inputFrequencyData!=null||(this.inputFrequencyData=new Uint8Array(this.input.analyser.frequencyBinCount)),this.input.analyser.getByteFrequencyData(this.inputFrequencyData),this.inputFrequencyData),this.getOutputByteFrequencyData=()=>(this.outputFrequencyData!=null||(this.outputFrequencyData=new Uint8Array(this.output.analyser.frequencyBinCount)),this.output.analyser.getByteFrequencyData(this.outputFrequencyData),this.outputFrequencyData),this.getInputVolume=()=>this.calculateVolume(this.getInputByteFrequencyData()),this.getOutputVolume=()=>this.calculateVolume(this.getOutputByteFrequencyData()),this.sendFeedback=e=>{this.canSendFeedback?(this.connection.sendMessage({type:"feedback",score:e?"like":"dislike",event_id:this.currentEventId}),this.lastFeedbackEventId=this.currentEventId,this.updateCanSendFeedback()):console.warn(this.lastFeedbackEventId===0?"Cannot send feedback: the conversation has not started yet.":"Cannot send feedback: feedback has already been sent for the current response.")},this.sendContextualUpdate=e=>{this.connection.sendMessage({type:"contextual_update",text:e})},this.sendUserMessage=e=>{this.connection.sendMessage({type:"user_message",text:e})},this.sendUserActivity=()=>{this.connection.sendMessage({type:"user_activity"})},this.options=t,this.connection=a,this.input=s,this.output=i,this.wakeLock=o,this.options.onConnect({conversationId:a.conversationId}),this.connection.onDisconnect(this.endSessionWithDetails),this.connection.onMessage(this.onMessage),this.input.worklet.port.onmessage=this.onInputWorkletMessage,this.output.worklet.port.onmessage=this.onOutputWorkletMessage,this.updateStatus("connected")}}export{O as R,L as f};
