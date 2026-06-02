export default function liquidRecorder( {
	maxDurationSeconds = null,
	maxSizeMb = null,
} = {} ) {
	return {
		state: 'idle', // 'idle' | 'recording' | 'recorded'
		elapsed: 0,
		duration: 0,
		error: '',
		isPlaying: false,
		maxDurationSeconds,
		maxSizeMb,

		_stream: null,
		_recorder: null,
		_chunks: [],
		_timerId: null,
		_audioContext: null,
		_analyser: null,
		_dataArray: null,
		_rafId: null,
		_audioUrl: null,
		_wavesurfer: null,
		_recorderStartedAt: 0,

		get recording() {
			return this.state === 'recording';
		},

		get hasRecording() {
			return this.state === 'recorded';
		},

		async toggle() {
			if ( this.state === 'idle' ) {
				await this.start();
			} else if ( this.state === 'recording' ) {
				this.stop();
			}
		},

		async start() {
			this.error = '';

			if ( !navigator.mediaDevices || typeof window.MediaRecorder === 'undefined' ) {
				this.error = 'Recording is not supported in this browser.';
				return;
			}

			try {
				this._stream = await navigator.mediaDevices.getUserMedia( { audio: true } );
			} catch ( e ) {
				this.error = e.name === 'NotAllowedError'
					? 'Microphone access was denied.'
					: `Microphone error: ${ e.message || e.name }`;
				return;
			}

			const mimeType = this._pickMimeType();
			this._chunks = [];
			this._recorder = new MediaRecorder( this._stream, mimeType ? { mimeType } : {} );

			this._recorder.addEventListener( 'dataavailable', ( e ) => {
				if ( e.data && e.data.size > 0 ) this._chunks.push( e.data );
			} );
			this._recorder.addEventListener( 'stop', () => this._finalize() );
			this._recorder.addEventListener( 'error', ( e ) => {
				this.error = `Recording error: ${ e.error?.message || 'unknown' }`;
				this._cleanupStream();
			} );

			this._recorder.start();
			this._recorderStartedAt = performance.now();
			this.state = 'recording';
			this.elapsed = 0;

			this._timerId = setInterval( () => {
				this.elapsed = ( performance.now() - this._recorderStartedAt ) / 1000;
				if ( this.maxDurationSeconds && this.elapsed >= this.maxDurationSeconds ) {
					this.stop();
				}
			}, 100 );

			this._setupLiveWaveform();
		},

		stop() {
			if ( !this._recorder || this._recorder.state === 'inactive' ) return;
			this._recorder.stop();
		},

		clear() {
			this._tearDownPlayback();
			this._cleanupStream();
			this._teardownLiveWaveform();

			if ( this._timerId ) {
				clearInterval( this._timerId );
				this._timerId = null;
			}

			if ( this.$refs.fileInput ) {
				const dt = new DataTransfer();
				this.$refs.fileInput.files = dt.files;
				// Programmatic .files assignment doesn't fire `change`; emit one so
				// any parent listener (e.g. <x-forms.recorder @change="...">) reacts.
				this.$refs.fileInput.dispatchEvent( new Event( 'change', { bubbles: true } ) );
			}

			this._chunks = [];
			this.elapsed = 0;
			this.duration = 0;
			this.error = '';
			this.isPlaying = false;
			this.state = 'idle';
		},

		togglePlayback() {
			if ( !this._wavesurfer ) return;
			this._wavesurfer.playPause();
		},

		async _finalize() {
			if ( this._timerId ) {
				clearInterval( this._timerId );
				this._timerId = null;
			}

			this._teardownLiveWaveform();
			this._cleanupStream();

			const recordedMime = this._recorder?.mimeType || 'audio/webm';
			const recordedBlob = new Blob( this._chunks, { type: recordedMime } );

			// fal.ai VEED Fabric accepts mp3/wav/m4a/aac/ogg, but mp4-container
			// audio (Chrome/Safari MediaRecorder default) is ambiguous to libmagic
			// and inconsistently accepted. Always transcode to mp3 so what we
			// upload is exactly what fal.ai documents as supported.
			let blob;
			try {
				blob = await this._transcodeToMp3( recordedBlob );
			} catch ( e ) {
				this.error = `Could not encode recording: ${ e.message || e.name || 'unknown' }`;
				this.state = 'idle';
				this._chunks = [];
				return;
			}

			const validationError = this._validate( blob );
			if ( validationError ) {
				this.error = validationError;
				this.state = 'idle';
				this._chunks = [];
				return;
			}

			const file = new File( [ blob ], `recording-${ Date.now() }.mp3`, { type: 'audio/mpeg' } );
			const dt = new DataTransfer();
			dt.items.add( file );
			if ( this.$refs.fileInput ) {
				this.$refs.fileInput.files = dt.files;
				// Programmatic .files assignment doesn't fire `change`; emit one so
				// any parent listener (e.g. <x-forms.recorder @change="...">) reacts.
				this.$refs.fileInput.dispatchEvent( new Event( 'change', { bubbles: true } ) );
			}

			this._audioUrl = URL.createObjectURL( blob );
			this.duration = this.elapsed;
			this.state = 'recorded';

			this.$nextTick( () => this._initWavesurfer() );
		},

		_initWavesurfer() {
			if ( typeof window.WaveSurfer === 'undefined' ) return;
			const container = this.$refs.playbackContainer;
			if ( !container ) return;

			while ( container.firstElementChild ) container.firstElementChild.remove();

			const styles = getComputedStyle( document.documentElement );
			const tokenColor = ( name, fallback ) => {
				const value = styles.getPropertyValue( name ).trim();
				return value ? `hsl(${ value })` : fallback;
			};

			this._wavesurfer = window.WaveSurfer.create( {
				container,
				waveColor: tokenColor( '--label', '#bcbac8' ),
				progressColor: tokenColor( '--primary', '#320580' ),
				cursorWidth: 0,
				barWidth: 1,
				interact: true,
				autoCenter: false,
				hideScrollbar: true,
				height: 22,
			} );
			this._wavesurfer.load( this._audioUrl );
			this._wavesurfer.on( 'play', () => { this.isPlaying = true; } );
			this._wavesurfer.on( 'pause', () => { this.isPlaying = false; } );
			this._wavesurfer.on( 'finish', () => { this.isPlaying = false; } );
			this._wavesurfer.on( 'ready', () => {
				const real = this._wavesurfer.getDuration();
				if ( Number.isFinite( real ) && real > 0 ) {
					this.duration = real;
				}
			} );
		},

		_tearDownPlayback() {
			if ( this._wavesurfer ) {
				try { this._wavesurfer.destroy(); } catch ( e ) { /* ignore */ }
				this._wavesurfer = null;
			}
			if ( this._audioUrl ) {
				URL.revokeObjectURL( this._audioUrl );
				this._audioUrl = null;
			}
		},

		_setupLiveWaveform() {
			const canvas = this.$refs.liveCanvas;
			if ( !canvas || !this._stream ) return;

			const AudioCtx = window.AudioContext || window.webkitAudioContext;
			if ( !AudioCtx ) return;

			this._audioContext = new AudioCtx();
			const source = this._audioContext.createMediaStreamSource( this._stream );
			this._analyser = this._audioContext.createAnalyser();
			this._analyser.fftSize = 2048;
			source.connect( this._analyser );

			this._dataArray = new Uint8Array( this._analyser.fftSize );

			const styles = getComputedStyle( document.documentElement );
			const stroke = styles.getPropertyValue( '--primary' ).trim();
			const strokeColor = stroke ? `hsl(${ stroke })` : '#320580';

			const draw = () => {
				if ( this.state !== 'recording' || !this._analyser ) return;
				this._rafId = requestAnimationFrame( draw );

				this._analyser.getByteTimeDomainData( this._dataArray );

				const dpr = window.devicePixelRatio || 1;
				const width = canvas.offsetWidth * dpr;
				const height = canvas.offsetHeight * dpr;
				if ( canvas.width !== width ) canvas.width = width;
				if ( canvas.height !== height ) canvas.height = height;

				const ctx = canvas.getContext( '2d' );
				ctx.clearRect( 0, 0, width, height );
				ctx.lineWidth = 1.5 * dpr;
				ctx.strokeStyle = strokeColor;
				ctx.beginPath();

				const sliceWidth = width / this._dataArray.length;
				let x = 0;
				for ( let i = 0; i < this._dataArray.length; i++ ) {
					const v = this._dataArray[ i ] / 128.0;
					const y = ( v * height ) / 2;
					if ( i === 0 ) ctx.moveTo( x, y );
					else ctx.lineTo( x, y );
					x += sliceWidth;
				}
				ctx.lineTo( width, height / 2 );
				ctx.stroke();
			};
			draw();
		},

		_teardownLiveWaveform() {
			if ( this._rafId ) {
				cancelAnimationFrame( this._rafId );
				this._rafId = null;
			}
			if ( this._audioContext ) {
				try { this._audioContext.close(); } catch ( e ) { /* ignore */ }
				this._audioContext = null;
			}
			this._analyser = null;
			this._dataArray = null;
		},

		_cleanupStream() {
			if ( this._stream ) {
				this._stream.getTracks().forEach( ( t ) => t.stop() );
				this._stream = null;
			}
		},

		_pickMimeType() {
			// Quality-ordered list of capture formats. Whatever the browser
			// produces, _finalize transcodes to mp3 before submit, so the only
			// thing that matters here is recording fidelity.
			const candidates = [
				'audio/webm;codecs=opus',
				'audio/webm',
				'audio/mp4',
				'audio/mpeg',
			];
			return candidates.find( ( m ) => window.MediaRecorder.isTypeSupported && window.MediaRecorder.isTypeSupported( m ) );
		},

		_extensionFor( mimeType ) {
			if ( mimeType.includes( 'mp3' ) || mimeType.includes( 'mpeg' ) ) return 'mp3';
			if ( mimeType.includes( 'wav' ) ) return 'wav';
			if ( mimeType.includes( 'mp4' ) ) return 'm4a';
			if ( mimeType.includes( 'aac' ) ) return 'aac';
			if ( mimeType.includes( 'webm' ) ) return 'webm';
			return 'audio';
		},

		async _transcodeToMp3( blob ) {
			if ( typeof window.lamejs === 'undefined' || !window.lamejs.Mp3Encoder ) {
				throw new Error( 'MP3 encoder not loaded' );
			}
			const AudioCtx = window.AudioContext || window.webkitAudioContext;
			if ( !AudioCtx ) {
				throw new Error( 'Web Audio API not supported' );
			}

			const arrayBuffer = await blob.arrayBuffer();
			const ctx = new AudioCtx();
			let audioBuffer;
			try {
				audioBuffer = await ctx.decodeAudioData( arrayBuffer );
			} finally {
				try { await ctx.close(); } catch ( e ) { /* ignore */ }
			}

			return this._encodeMp3( audioBuffer );
		},

		_encodeMp3( audioBuffer ) {
			// Mix down to mono — voice intelligibility doesn't need stereo and
			// mono cuts encoded size in half.
			const length = audioBuffer.length;
			const numChannels = audioBuffer.numberOfChannels;
			const pcmFloat = new Float32Array( length );
			if ( numChannels === 1 ) {
				pcmFloat.set( audioBuffer.getChannelData( 0 ) );
			} else {
				const left = audioBuffer.getChannelData( 0 );
				const right = audioBuffer.getChannelData( 1 );
				for ( let i = 0; i < length; i++ ) {
					pcmFloat[ i ] = ( left[ i ] + right[ i ] ) * 0.5;
				}
			}

			// Float32 [-1, 1] → Int16 PCM, what lamejs expects.
			const pcmInt16 = new Int16Array( length );
			for ( let i = 0; i < length; i++ ) {
				const s = Math.max( -1, Math.min( 1, pcmFloat[ i ] ) );
				pcmInt16[ i ] = s < 0 ? s * 0x8000 : s * 0x7FFF;
			}

			// 1 channel, 128 kbps — plenty for voice, ~1 MB/min upload.
			const encoder = new window.lamejs.Mp3Encoder( 1, audioBuffer.sampleRate, 128 );
			const blockSize = 1152;
			const mp3Chunks = [];

			for ( let i = 0; i < pcmInt16.length; i += blockSize ) {
				const chunk = pcmInt16.subarray( i, i + blockSize );
				const encoded = encoder.encodeBuffer( chunk );
				if ( encoded.length > 0 ) mp3Chunks.push( encoded );
			}

			const flushed = encoder.flush();
			if ( flushed.length > 0 ) mp3Chunks.push( flushed );

			return new Blob( mp3Chunks, { type: 'audio/mpeg' } );
		},

		_validate( blob ) {
			if ( this.maxSizeMb && blob.size > this.maxSizeMb * 1024 * 1024 ) {
				return `Recording too large (max ${ this.maxSizeMb } MB).`;
			}
			return null;
		},

		formatDuration( seconds ) {
			const total = Math.max( 0, Math.floor( seconds ) );
			const m = Math.floor( total / 60 );
			const s = total % 60;
			return `${ m }:${ s < 10 ? '0' : '' }${ s }`;
		},

		destroy() {
			this._tearDownPlayback();
			this._cleanupStream();
			this._teardownLiveWaveform();
			if ( this._timerId ) clearInterval( this._timerId );
		},
	};
}
