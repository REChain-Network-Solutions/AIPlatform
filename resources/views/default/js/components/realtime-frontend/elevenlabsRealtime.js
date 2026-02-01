import { Player } from './player.js';
import { Recorder } from './recorder.js';
import { Conversation } from '@11labs/client';
import { Alpine } from '~vendor/livewire/livewire/dist/livewire.esm';

Alpine.store( 'realtimeChatStatus', {
	active: false,
	conversationStarted: false,

	setActive( value ) {
		this.active = value;

		this.onActiveChange();
	},

	setConversationStarted( value ) {
		this.conversationStarted = value;

		this.onConversationStartedChange();
	},

	onActiveChange() {
		document.querySelectorAll( '.lqd-realtime-chat-button' ).forEach( button => button.classList.toggle( 'active', this.active ) );
		document.querySelector( '.lqd-audio-vis-wrap' )?.classList?.toggle( 'active', this.active );
	},

	onConversationStartedChange() {
		const chatsWrapper = document.querySelector( '.chats-wrap' );

		chatsWrapper.classList.toggle( 'conversation-started', this.conversationStarted );
		chatsWrapper.classList.toggle( 'conversation-not-started', !this.conversationStarted );

		document.querySelectorAll( '.lqd-realtime-chat-button' ).forEach( button => {
			button.classList.toggle( 'conversation-started', this.conversationStarted );
			button.classList.toggle( 'conversation-not-started', !this.conversationStarted );
		} );
	},
} );

export default agent_id => ( {
	/**@type {String} */
	agentId: agent_id,
	recordingActive: false,
	buffer: new Uint8Array(),
	/** @type {Conversation} */
	conversation: null,
	/** @type {Recorder} */
	audioRecorder: null,
	/** @type {Player} */
	audioPlayer: null,
	/** @type {'idle' | 'loading' | 'recording' | 'playing'} */
	activeVisulaizer: 'idle',
	/** @type {HTMLElement} */
	audioVisWrap: null,
	/** @type {HTMLElement} */
	audioVisBars: null,
	/** @type {HTMLElement} */
	audioVisDotWrap: null,
	/** @type {HTMLElement} */
	audioVisLoader: null,
	conversationArea: document.querySelector( '.conversation-area' ),
	chatsContainer: document.querySelector( '.chats-container' ),
	/** @type {HTMLTemplateElement} */
	userBubbleTemplate: document.querySelector( '#chat_user_bubble' ),
	/** @type {HTMLTemplateElement} */
	aiBubbleTemplate: document.querySelector( '#chat_ai_bubble' ),
	lastAiBubble: null,
	lastUserBubble: null,
	lastUserQuestion: '',
	lastAiResponse: '',
	lastResponseSaved: false,

	init() {
		this.audioVisWrap = document.querySelector( '.lqd-audio-vis-wrap' );
		this.audioVisBars = this.audioVisWrap?.querySelectorAll( '.lqd-audio-vis-bar' );
		this.audioVisDotWrap = this.audioVisWrap?.querySelector( '.lqd-audio-vis-dot-wrap' );
		this.audioVisLoader = this.audioVisWrap?.querySelector( '.lqd-audio-vis-loader' );

		this.processAudioRecordingBuffer = this.processAudioRecordingBuffer.bind( this );
	},
	async start() {
		if ( Alpine.store( 'realtimeChatStatus' ).isActive ) return;

		Alpine.store( 'realtimeChatStatus' ).setActive( true );
		this.switchVisualizers( 'waiting' );

		const result = await this.checkBalance( true );

		if ( result.shouldStop ) {
			toastr.error( result.errorMsg );
			this.stop();
			return;
		}

		this.conversation = await Conversation.startSession( {
			agentId: this.agentId,
			onConnect: async () => {
				await Promise.all( [ this.startRecorder(), this.startPlayer() ] )
					.then( () => {
						this.startBarsVisualizer();
						this.startDotVisualizer();

						this.switchVisualizers( 'idle' );
						Alpine.store( 'realtimeChatStatus' ).setConversationStarted( true );
					} )
					.catch( error => {
						this.stop();
						console.error( 'Error starting recorder and player:', error );
						this.appendToChatBubble( 'ai', '[Error]: Unable to start audio recorder and player. Please check your microphone permissions and refresh the page.' );
					} );
			},
			onDisconnect: () => {
				this.disconnectHandle( this.conversation?.connection );
				this.stop();
			},
			onModeChange: mode => {
				this.checkBalance().then( result => {
					if ( result.shouldStop ) {
						this.stop();
						toastr.error( result.errorMsg );
						return;
					}
				} );

			},
			onMessage: message => {
				if ( message.source == 'ai' ) {
					this.createChatBubble( 'ai' );
					this.appendToChatBubble( 'ai', message.message );
					this.lastAiResponse = message.message;

					if ( 'saveResponseAsync' in window ) {
						saveResponseAsync(
							this.lastUserQuestion.trim(),
							this.lastAiResponse.trim(),
							document.querySelector( '#chat_id' ).value,
							'',
							'',
							'',
							'',
							'elevenlabs-voice-chatbot'
						).then(result => {
							if (result && result.message.id) {
								this.changeVoiceChatTitle(result.message.id);
							}
						});

						this.lastResponseSaved = true;
					}

					if ( 'formatString' in window && this.lastAiBubble ) {
						this.lastAiBubble.innerHTML = formatString( this.lastAiResponse );
					}

					this.lastAiResponse = this.lastUserQuestion = '';
				} else {
					this.createChatBubble( 'user' );
					this.appendToChatBubble( 'user', message.message );
					this.lastUserQuestion += ' ' + message.message;

					this.lastResponseSaved = false;
				}

			},
			onError: error => {
				console.error( 'Error:', error );
				this.stop();
			}
		} );
	},
	changeVoiceChatTitle(messageId) {
		const chat = document.querySelector(`#chat_${document.querySelector('#chat_id').value}`);
		const chatTitleEl = chat?.querySelector('.chat-item-title');

		if (!chatTitleEl) return;

		$.ajax({
			type: 'post',
			url: '/dashboard/change-chat-title',
			data: {
				streamed_message_id: messageId,
			},
			success: function (data) {
				if (data.changed) {
					const newTitle = data.new_title.replaceAll(' ', '\u00a0');
					const newTitleStringArray = newTitle.split('');

					chatTitleEl.innerText = '';

					const interval = setInterval(() => {
						chatTitleEl.innerText += newTitleStringArray.shift();

						if (!newTitleStringArray.length) {
							clearInterval(interval);
						}
					}, 30);
				}
			},
			error: function(error) {
				console.error('Error changing chat title:', error);
			}
		});
	},
	async stop() {
		if ( !this.lastResponseSaved && 'saveResponseAsync' in window && this.lastUserQuestion.trim() !== '' && this.lastAiResponse.trim() !== '' ) {
			saveResponseAsync(
				this.lastUserQuestion.trim(),
				this.lastAiResponse.trim(),
				document.querySelector( '#chat_id' ).value,
				'',
				'',
				'',
				'',
				'elevenlabs-voice-chatbot'
			);
		}

		this.resetPlayers();
		if ( this.conversation ) {
			await this.conversation.endSession();
			this.conversation = null;
		}

		this.switchVisualizers( '' );

		Alpine.store( 'realtimeChatStatus' ).setActive( false );
	},
	async startPlayer() {
		try {
			this.audioPlayer = new Player();
			await this.audioPlayer.init( 24000 );
		} catch ( error ) {
			console.error( 'Error starting audio player:', error );
		}
	},
	async startRecorder() {
		try {
			this.audioRecorder = new Recorder( this.processAudioRecordingBuffer );
			const stream = await navigator.mediaDevices.getUserMedia( { audio: true, video: false } );

			await this.audioRecorder.start( stream );
			this.recordingActive = true;
		} catch ( error ) {
			console.error( 'Error starting audio recorder:', error );
		}
	},
	combineArray( newData ) {
		const newBuffer = new Uint8Array( this.buffer.length + newData.length );
		newBuffer.set( this.buffer );
		newBuffer.set( newData, this.buffer.length );
		this.buffer = newBuffer;
	},
	processAudioRecordingBuffer( data ) {
		const uint8Array = new Uint8Array( data );

		this.combineArray( uint8Array );

		if ( this.buffer.length >= 4800 ) {
			const toSend = new Uint8Array( this.buffer.slice( 0, 4800 ) );
			this.buffer = new Uint8Array( this.buffer.slice( 4800 ) );
			const regularArray = String.fromCharCode( ...toSend );
			const base64 = btoa( regularArray );
		}
	},
	async resetPlayers() {
		this.recordingActive = false;

		this.audioRecorder?.stop();
		this.audioPlayer?.clear();
	},
	getSystemMessage() {
		return '';
	},
	getTemperature() {
		return parseFloat( 0.8 );
	},
	getVoice() {
		// alloy, echo, or shimmer
		return 'alloy';
	},
	switchVisualizers( activeVisulaizer ) {
		this.activeVisulaizer = activeVisulaizer;

		this.audioVisWrap?.setAttribute( 'data-state', this.activeVisulaizer );
	},
	createChatBubble( role ) {
		const template = role === 'user' ? this.userBubbleTemplate : this.aiBubbleTemplate;
		const bubble = template.content.cloneNode( true );
		const bubbleContainer = bubble.querySelector( '.chat-content' );

		this.chatsContainer.appendChild( bubble );

		if ( role === 'user' ) {
			this.lastUserBubble = bubbleContainer;
		} else {
			this.lastAiBubble = bubbleContainer;
		}

		this.scrollConversationAreaToBottom();
	},
	appendToChatBubble( role, text ) {
		const bubble = role === 'user' ? this.lastUserBubble : this.lastAiBubble;

		if ( bubble ) {
			bubble.textContent += text;

			this.scrollConversationAreaToBottom();
		} else {
			this.createChatBubble( role );
			this.appendToChatBubble( role, text );
		}
	},
	scrollConversationAreaToBottom() {
		this.conversationArea.scrollTo( {
			top: this.conversationArea.scrollHeight + 200,
			left: 0
		} );
	},
	startBarsVisualizer() {
		if ( !this.audioVisBars?.length ) return;

		const audioAnalyser = this.audioPlayer.audioContext.createAnalyser();
		audioAnalyser.fftSize = 4096;

		const bufferLength = audioAnalyser.frequencyBinCount;
		const dataArray = new Uint8Array( bufferLength );
		const barCount = this.audioVisBars.length;

		this.audioPlayer.playbackNode.connect( audioAnalyser );

		// Define frequency ranges for each bar (in Hz)
		const frequencyRanges = [
			[ 85, 150 ],   // Low
			[ 150, 250 ],  // Low-mid
			[ 250, 400 ],  // Mid
			[ 400, 600 ],  // Mid-high
			[ 600, 1000 ]  // High (including some overtones)
		];

		// Create an array to store the current heights of bars
		this.barHeights = this.barHeights || new Array( barCount ).fill( 0 );

		const animate = () => {
			audioAnalyser.getByteFrequencyData( dataArray );

			this.audioVisBars.forEach( ( bar, index ) => {
				const [ lowFreq, highFreq ] = frequencyRanges[ index ];

				// Convert frequency to FFT bin index
				const lowIndex = Math.floor( lowFreq / ( this.audioPlayer.audioContext.sampleRate / audioAnalyser.fftSize ) );
				const highIndex = Math.ceil( highFreq / ( this.audioPlayer.audioContext.sampleRate / audioAnalyser.fftSize ) );

				// Get the maximum amplitude in this frequency range
				let maxAmplitude = 0;
				for ( let i = lowIndex; i <= highIndex && i < dataArray.length; i++ ) {
					if ( dataArray[ i ] > maxAmplitude ) {
						maxAmplitude = dataArray[ i ];
					}
				}

				// Calculate target height (0-80)
				let targetHeight = ( maxAmplitude / 255 ) * 80;

				// Smooth the movement
				this.barHeights[ index ] += ( targetHeight - this.barHeights[ index ] ) * 0.4;

				// Add some randomness for natural look
				this.barHeights[ index ] += ( Math.random() - 0.5 ) * 2;

				// Ensure height is between 5% and 100%
				this.barHeights[ index ] = Math.max( 5, Math.min( 100, this.barHeights[ index ] ) );

				// Animate the bar height
				bar.animate(
					[
						{ height: bar.style.height },
						{ height: `${ this.barHeights[ index ] }%` }
					],
					{
						duration: 30,
						fill: 'forwards',
						easing: 'linear'
					}
				);
			} );

			requestAnimationFrame( animate );
		};

		animate();
	},
	startDotVisualizer() {
		if ( !this.audioRecorder || !this.audioVisDotWrap ) return;

		const analyser = this.audioRecorder.audioContext.createAnalyser();
		analyser.fftSize = 256;
		const bufferLength = analyser.frequencyBinCount;
		const dataArray = new Uint8Array( bufferLength );

		this.audioRecorder.getMediaStreamSource().connect( analyser );

		const dot = this.audioVisDotWrap.querySelector( '.lqd-audio-vis-dot' );

		if ( !dot ) return;

		const animate = () => {
			analyser.getByteFrequencyData( dataArray );

			let sum = 0;
			for ( let i = 0; i < bufferLength; i++ ) {
				sum += dataArray[ i ];
			}
			const average = sum / bufferLength;

			const scale = 1 + ( average / 256 ) * 1.5;
			const opacity = Math.max( 0.2, 1 - ( scale - 1 ) / 1.5 ); // Minimum opacity of 0.2

			dot.style.transform = `scale(${ scale })`;
			dot.style.opacity = opacity.toFixed( 2 ); // Limit to two decimal places

			requestAnimationFrame( animate );
		};

		animate();
	},
	checkBalance( onStart = false ) {
		return new Promise( resolve => {
			$.ajax( {
				url: '/dashboard/admin/voice-chatbot/check-balance',
				type: 'POST',
				data: {
					onStart: onStart,
				},
				dataType: 'json',
				success: response => {
					const shouldStop = response.status !== 'success';
					const errorMsg = response.message || '';
					resolve( { shouldStop, errorMsg } );
				},
				error: () => {
					const shouldStop = true;
					const errorMsg = 'An error occurred.';
					resolve( { shouldStop, errorMsg } );
				}
			} );
		} );
	},
	disconnectHandle( connection ) {
		if ( connection?.disconnectionDetails?.reason == 'error' ) {
			toastr.error( 'Something went wrong. Please contact support for assistance' );
			console.error( connection?.disconnectionDetails?.message );
		}
	}
} );
