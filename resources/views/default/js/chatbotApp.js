// import './bootstrap';
import {Alpine, Livewire} from '~vendor/livewire/livewire/dist/livewire.esm';
import ajax from '~nodeModules/@imacrayon/alpine-ajax';
import {fetchEventSource} from '@microsoft/fetch-event-source';
// import modal from "./components/modal";
import clipboard from "./components/clipboard";
import '../scss/chatbot-embed.scss';

window.fetchEventSource = fetchEventSource;
const alpine = window.Alpine || Alpine;
window.Alpine = alpine;
const livewire = window.Livewire || Livewire;
window.Livewire = livewire;

alpine.plugin(ajax);
console.log('chatbotApp yüklendi');

document.addEventListener('alpine:init', () => {
    alpine.data('clipboard', (data) => clipboard(data));
});

if ( !window.__livewireStarted ) {
    window.__livewireStarted = true;
    livewire.start();
}

document.querySelectorAll('[magic-load]').forEach(function (element) {
    element.removeAttribute('magic-load');
});
