@if(\App\Helpers\Classes\Helper::showIntercomForVipMembership())
	<script>
		window.intercomSettings = {
			api_base: "https://api-iam.intercom.io",
			app_id: "udcwpqev",
		};
	</script>

	<script>
		(function(){var w=window;var ic=w.Intercom;if(typeof ic==="function"){ic('reattach_activator');ic('update',w.intercomSettings);}else{var d=document;var i=function(){i.c(arguments);};i.q=[];i.c=function(args){i.q.push(args);};w.Intercom=i;var l=function(){var s=d.createElement('script');s.type='text/javascript';s.async=true;s.src='https://widget.intercom.io/widget/udcwpqev';var x=d.getElementsByTagName('script')[0];x.parentNode.insertBefore(s,x);};if(document.readyState==='complete'){l();}else if(w.attachEvent){w.attachEvent('onload',l);}else{w.addEventListener('load',l,false);}}})();
	</script>
@else
	@includeWhen((auth()->user()?->isAdmin() && $app_is_not_demo && ! setting('hide_premium_support_chat', 0)), 'default.panel.layout.premium.widget')
@endif

