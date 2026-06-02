export default function liquidDropArea( {
	multiple = false,
	accept = '',
	maxSizeMb = null,
} = {} ) {
	const urlCache = new Map();

	const releaseUrl = ( file ) => {
		if ( urlCache.has( file ) ) {
			URL.revokeObjectURL( urlCache.get( file ) );
			urlCache.delete( file );
		}
	};

	return {
		files: [],
		dragOver: false,
		error: '',
		multiple,
		accept,
		maxSizeMb,

		handleFileSelect( event ) {
			this.addFiles( event.target.files );
		},

		handleFileDrop( event ) {
			this.dragOver = false;
			this.addFiles( event.dataTransfer.files );
		},

		addFiles( fileList ) {
			this.error = '';

			const incoming = Array.from( fileList );
			if ( incoming.length === 0 ) return;

			let pool;
			if ( this.multiple ) {
				pool = [ ...this.files ];
			} else {
				this.files.forEach( releaseUrl );
				pool = [];
			}

			for ( const file of incoming ) {
				const err = this.validate( file );
				if ( err ) {
					this.error = err;
					continue;
				}

				const isDup = pool.some(
					( f ) => f.name === file.name
						&& f.size === file.size
						&& f.lastModified === file.lastModified,
				);
				if ( isDup ) continue;

				pool.push( file );
			}

			this.files = pool;
			this.syncInput();
		},

		removeFile( index ) {
			const file = this.files[ index ];
			if ( file ) releaseUrl( file );
			this.files = this.files.filter( ( _, i ) => i !== index );
			this.error = '';
			this.syncInput();
		},

		clear() {
			this.files.forEach( releaseUrl );
			this.files = [];
			this.error = '';
			this.syncInput();
		},

		syncInput() {
			const input = this.$refs.fileInput;
			if ( !input ) return;
			const dt = new DataTransfer();
			this.files.forEach( ( f ) => dt.items.add( f ) );
			input.files = dt.files;
		},

		validate( file ) {
			if ( !this.matchesAccept( file ) ) {
				return `${ file.name }: file type not allowed.`;
			}
			if ( this.maxSizeMb && file.size > this.maxSizeMb * 1024 * 1024 ) {
				return `${ file.name }: file too large (max ${ this.maxSizeMb } MB).`;
			}
			return null;
		},

		matchesAccept( file ) {
			if ( !this.accept ) return true;
			const accepts = this.accept.split( ',' ).map( ( s ) => s.trim() ).filter( Boolean );
			if ( accepts.length === 0 ) return true;

			return accepts.some( ( accept ) => {
				if ( accept.startsWith( '.' ) ) {
					return file.name.toLowerCase().endsWith( accept.toLowerCase() );
				}
				if ( accept.endsWith( '/*' ) ) {
					const prefix = accept.slice( 0, -1 );
					return ( file.type || '' ).startsWith( prefix );
				}
				return file.type === accept;
			} );
		},

		previewType( file ) {
			if ( !file?.type ) return 'file';
			if ( file.type.startsWith( 'image/' ) ) return 'image';
			if ( file.type.startsWith( 'video/' ) ) return 'video';
			if ( file.type.startsWith( 'audio/' ) ) return 'audio';
			return 'file';
		},

		previewUrl( file ) {
			if ( !urlCache.has( file ) ) {
				urlCache.set( file, URL.createObjectURL( file ) );
			}
			return urlCache.get( file );
		},

		fileSize( bytes ) {
			if ( !bytes ) return '';
			const units = [ 'B', 'KB', 'MB', 'GB' ];
			let size = bytes;
			let i = 0;
			while ( size >= 1024 && i < units.length - 1 ) {
				size /= 1024;
				i++;
			}
			return `${ size.toFixed( size < 10 && i > 0 ? 1 : 0 ) } ${ units[ i ] }`;
		},
	};
}
