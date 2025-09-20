( async function () {
	const scriptTag = document.currentScript;
	const url = new URL( scriptTag.getAttribute( 'src' ) );
	const chatbotHostOrigin = `${ url.origin }`;
	const chatBotUuid = scriptTag.getAttribute( 'data-chatbot-uuid' );
	const iFrameUrl = `${ chatbotHostOrigin }/chatbot-voice/${ chatBotUuid }/frame`;

	const widgetMarkup = `
<div id="lqd-ext-voice-chatbot-wrapper">
	<style>
		#lqd-ext-voice-chatbot-wrapper {
			position: fixed;
			display: flex;
			align-items: center;
			justify-content: center;
			width: var(--lqd-ext-voice-window-w);
			height: var(--lqd-ext-voice-window-h);
			bottom: var(--lqd-ext-voice-offset-y, 30px);
			left: var(--lqd-ext-voice-offset-x, 30px);
			border-radius: var(--lqd-ext-voice-box-radius);
			z-index: 99999;

			opacity: 0;
            visibility: hidden;
			transition: all 0.1s;
		}

		#lqd-ext-voice-chatbot-wrapper[data-ready=true] {
			opacity: 1;
            visibility: visible;
		}

		#lqd-ext-voice-chatbot-wrapper[frame-x-pos=right] {
			left: auto;
			right: var(--lqd-ext-voice-offset-x, 30px);
		}

		#lqd-ext-voice-chatbot-iframe {
            width: 100%;
            height: 100%;
        }

		.lqd-ext-voice-chatbot-not-loaded {
            margin: 0;
            padding: 1rem;
        }
	</style>
	${ iFrameUrl ? `
		<iframe
			src="${ iFrameUrl }"
			frameborder="0"
			allowfullscreen
			allowtransparency
			allow="microphone; camera; screen-wake-lock"
			id="lqd-ext-voice-chatbot-iframe"
			name="lqd-ext-voice-chatbot-iframe"
			crossOrigin="anonymous"
			onload="
				const wrapper = document.getElementById('lqd-ext-voice-chatbot-wrapper');
				window.addEventListener('message', event => {
					if ( event.origin !== '${ chatbotHostOrigin }' || event.data.type !== 'lqd-ext-voice-chatbot-response-styling' || !wrapper ) return;
					const { styles, attrs } = event.data.data;
					Object.entries(styles).forEach(([key, value]) => {
						wrapper.style.setProperty(key, value);
					});
					Object.entries(attrs).forEach(([key, value]) => {
						wrapper.setAttribute(key, value);
					});
					wrapper.setAttribute('data-ready', 'true');
				});

				this.contentWindow.postMessage({
					type: 'lqd-ext-voice-chatbot-request-styling',
				}, '${ chatbotHostOrigin }');
			"
		></iframe>` : `
	<p class="lqd-ext-voice-chatbot-not-loaded">Could not setup the voice chatbot</p>
	`}
</div>`;

	document.body.insertAdjacentHTML( 'beforeend', widgetMarkup );
} )();
