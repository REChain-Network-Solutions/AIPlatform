import { Alpine } from '~vendor/livewire/livewire/dist/livewire.esm';
import { Conversation } from '@11labs/client';
import { Recorder } from './components/realtime-frontend/recorder.js';

const elevenLabsConversationalAI = ( agentId, botUuid ) => ( {
	/**@type {String} */
	agentId: agentId,
	/**@type {String} */
	uuId: botUuid,
	/**@type {String} */
	bubbleMessage: 'Need help?',
	/**@type {Conversation} */
	conversation: null,
	/**@type {Recorder} */
	audioRecorder: null,
	/**@type {HTMLElement} */
	chatbotStatus: null,
	/**@type {HTMLElement} */
	startConversationBtn: null,
	/**@type {HTMLElement} */
	stopConversationBtn: null,
	/**@type {HTMLElement} */
	audioVisEl: null,
	/** @type {MediaStream|null} */
	audioStream: null,


	init() {
		this.chatbotStatus = document.getElementById( 'lqd-ext-chatbot-voice-bot-status' );
		this.bubbleMessage = this.chatbotStatus.textContent;
		this.startConversationBtn = document.getElementById( 'lqd-ext-chatbot-voice-start-btn' );
		this.stopConversationBtn = document.getElementById( 'lqd-ext-chatbot-voice-end-btn' );
		this.audioVisEl = document.getElementById( 'lqd-ext-chatbot-voice-vis-img' );

		this.initRecorder();
		this.addEventListeners();
	},
	// add event listeners
	addEventListeners() {
		this.startConversationBtn.addEventListener( 'click', () => {
			this.startConversation();
		} );
		this.stopConversationBtn.addEventListener( 'click', () => this.stopConversation() );
	},
	// start conversation
	async startConversation() {
		// disable the btn to prevent double click
		this.startConversationBtn.setAttribute( 'disabled', true );
		this.startConversationBtn.querySelector( 'span' ).textContent = 'starting...';

		const result = await this.checkVoiceBalance( true );
		if ( result.shouldStop ) {
			this.startConversationBtn.removeAttribute( 'disabled' );
			this.startConversationBtn.querySelector( 'span' ).textContent = 'Voice Chat';

			alert( result.errorMsg );
			return;
		}

		try {
			// request microphone permission
			const stream = await navigator.mediaDevices.getUserMedia( { audio: true, video: false } );
			this.audioStream = stream;
			this.conversation = await Conversation.startSession( {
				agentId: this.agentId,
				onConnect: async () => {
					this.updateUIByStatus( 'calling' );

					await this.audioRecorder?.start( stream );
					this.startDotVisualizer();
				},
				onDisconnect: () => {
					this.disconnectHandle( this.conversation?.connection );
					this.updateUIByStatus();
					this.storeConversation( this.conversation.getId() );
					this.audioRecorder?.stop();
					this.stopAudioStream();
				},
				onModeChange: mode => {
					this.chatbotStatus.textContent = mode.mode === 'speaking' ? 'speaking' : 'listening';
					this.checkVoiceBalance().then(result2 => {
						if ( result2.shouldStop ) {
							console.log(result2.shouldStop);
							this.updateUIByStatus();
							this.audioRecorder?.stop();
							this.stopAudioStream();
							this.stopConversation();
							alert( result2.errorMsg );
							return;
						}
					});
				},
				onError: error => {
					console.error( 'Error:', error );
				}
			} )
		} catch ( error ) {
			this.updateUIByStatus();
			alert( 'Something went wrong with voice agent' );
			console.error( error );
		}
	},
	// stop conversation
	async stopConversation() {
		if ( this.conversation ) {
			await this.conversation.endSession();
			this.conversation = null;
		}
	},
	// start recorder (this is needed for conversation status visualization)
	async initRecorder() {
		try {
			this.audioRecorder = new Recorder( this.handleAudioRecordingBuffer );
		} catch ( error ) {
			console.error( 'Error starting audio recorder:', error );
		}
	},
	// create conversation
	async storeConversation( conversationId ) {
		const res = await fetch( `/api/v2/chatbot-voice/${ this.uuId }/store-conversation`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Accept': 'application/json'
			},
			body: JSON.stringify( {
				'conversation_id': conversationId
			} )
		} );

		try {
			const resData = await res.json();

			if ( !res.ok ) {
				console.error( 'Failed create conversation:', resData.message );
			}
		} catch ( error ) {
			console.error( 'Failed parse JSON:', error );
		}
	},
	/**
	 * update ui by calling status
	 * @param {String} status
	 */
	updateUIByStatus( status = 'default' ) {
		if ( status == 'default' ) {
			this.startConversationBtn.style.display = 'flex';
			this.stopConversationBtn.style.display = 'none';
			this.chatbotStatus.textContent = this.bubbleMessage;

			// reset visualizer
			if ( this.audioVisEl ) {
				this.audioVisEl.style.transform = 'scale(1)';
				this.audioVisEl.style.opacity = 1;
			}

			this.startConversationBtn.removeAttribute( 'disabled' );
			this.startConversationBtn.querySelector( 'span' ).textContent = 'Voice Chat';
		} else if ( status == 'calling' ) {
			this.startConversationBtn.style.display = 'none';
			this.stopConversationBtn.style.display = 'flex';
		}
	},
	stopAudioStream() {
		if ( this.audioStream ) {
			this.audioStream.getTracks().forEach( track => track.stop() );
			this.audioStream = null;
		}
	},
	checkVoiceBalance( onStart = false ) {
		return new Promise( async ( resolve ) => {
			try {
				const response = await fetch( '/chatbot-voice/checkVoiceBalance', {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						'X-Requested-With': 'XMLHttpRequest',
						'X-CSRF-TOKEN': document.querySelector( 'meta[name="csrf-token"]' )?.content // if Laravel CSRF protection is on
					},
					body: JSON.stringify( {
						onStart: onStart,
						uuId: this.uuId
					} )
				} );

				if ( !response.ok ) {
					resolve( { shouldStop: true, errorMsg: 'An error occurred.' } );
					return;
				}

				const data = await response.json();
				const shouldStop = data.status === 'error';
				const errorMsg = data.message || '';
				resolve( { shouldStop, errorMsg } );

			} catch ( error ) {
				console.error( 'checkBalance fetch failed:', error );
				resolve( { shouldStop: true, errorMsg: 'An error occurred.' } );
			}
		} );
	},
	handleAudioRecordingBuffer( data ) { },
	startDotVisualizer() {
		if ( !this.audioRecorder || !this.audioVisEl ) return;

		const analyser = this.audioRecorder.audioContext.createAnalyser();
		analyser.fftSize = 256;
		const bufferLength = analyser.frequencyBinCount;
		const dataArray = new Uint8Array( bufferLength );

		this.audioRecorder.getMediaStreamSource().connect( analyser );

		if ( !this.audioVisEl ) return;

		const animate = () => {
			analyser.getByteFrequencyData( dataArray );

			let sum = 0;
			for ( let i = 0; i < bufferLength; i++ ) {
				sum += dataArray[ i ];
			}
			const average = sum / bufferLength;

			const scale = 1 + ( average / 256 ) * 1.2;
			const opacity = Math.max( 0.2, 1 - ( scale - 1 ) / 1.5 ); // Minimum opacity of 0.2

			this.audioVisEl.style.transform = `scale(${ scale })`;
			this.audioVisEl.style.opacity = opacity.toFixed( 2 ); // Limit to two decimal places

			requestAnimationFrame( animate );
		}

		animate();
	},
	disconnectHandle( connection ) {
		if ( connection?.disconnectionDetails?.reason == 'error' ) {
			alert( connection?.disconnectionDetails?.message || 'Something went wrong on agent' );
		}
	}
} );

window.Alpine = Alpine;
document.addEventListener( 'alpine:init', () => {
	Alpine.data( 'elevenLabsConversationalAI', elevenLabsConversationalAI );
} );
