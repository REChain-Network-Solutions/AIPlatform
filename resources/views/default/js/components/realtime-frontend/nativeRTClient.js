/**
 * Minimal native-WebSocket replacement for the (beta) rt-client library.
 * Matches the subset of LowLevelRTClient used by openaiRealtime.js:
 *   new NativeRTClient({ key }, { model })
 *   await client.send(obj)
 *   for await (const msg of client.messages()) { ... }
 *   client.close()
 *
 * Uses OpenAI Realtime GA subprotocols:
 *   ["realtime", "openai-insecure-api-key.<ephemeralKey>"]
 */
export class NativeRTClient {
	constructor( credential, options ) {
		const url = new URL( 'wss://api.openai.com/v1/realtime' );
		url.searchParams.set( 'model', options.model );

		this._queue = [];
		this._resolvers = [];
		this._closed = false;

		this._ready = new Promise( ( resolve, reject ) => {
			try {
				this._ws = new WebSocket( url.toString(), [
					'realtime',
					`openai-insecure-api-key.${ credential.key }`,
				] );
			} catch ( err ) {
				reject( err );

				return;
			}

			this._ws.addEventListener( 'open', () => resolve() );
			this._ws.addEventListener( 'error', e => {
				if ( !this._opened ) {
					reject( e );
				}
			} );
			this._ws.addEventListener( 'message', evt => {
				let data;

				try {
					data = JSON.parse( evt.data );
				} catch ( e ) {
					return;
				}

				if ( this._resolvers.length > 0 ) {
					this._resolvers.shift()( { value: data, done: false } );
				} else {
					this._queue.push( data );
				}
			} );
			this._ws.addEventListener( 'close', () => {
				this._closed = true;

				while ( this._resolvers.length > 0 ) {
					this._resolvers.shift()( { value: undefined, done: true } );
				}
			} );
		} );
	}

	async send( message ) {
		await this._ready;

		if ( this._closed || this._ws.readyState !== WebSocket.OPEN ) {
			throw new Error( 'Socket is not open' );
		}

		this._ws.send( JSON.stringify( message ) );
	}

	messages() {
		const self = this;

		return {
			[ Symbol.asyncIterator ]() {
				return {
					next() {
						if ( self._queue.length > 0 ) {
							return Promise.resolve( { value: self._queue.shift(), done: false } );
						}

						if ( self._closed ) {
							return Promise.resolve( { value: undefined, done: true } );
						}

						return new Promise( resolve => self._resolvers.push( resolve ) );
					},
					return() {
						self.close();

						return Promise.resolve( { value: undefined, done: true } );
					},
				};
			},
		};
	}

	close() {
		if ( this._ws && this._ws.readyState !== WebSocket.CLOSED ) {
			try {
				this._ws.close();
			} catch ( e ) {}
		}

		this._closed = true;
	}
}
