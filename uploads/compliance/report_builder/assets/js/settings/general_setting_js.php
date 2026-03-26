<script>

	(function($) {
		"use strict";  

		appValidateForm($('.general_setting'), {
			report_title: 'required',
			category_id: 'required',
			report_name: 'required',
			records_per_page: 'required',
		});

	})(jQuery); 

	// option-show
	$('#specific_staff').on('change', function() {
		'use strict';

		var input_name_status = $('input[id="specific_staff"]').is(":checked");
		if(input_name_status == true){
			$('.option-show').removeClass('hide');
			var is_public = $('input[id="is_public"]').is(":checked");

			if(is_public == true){
      			$('input[id="is_public"]').prop('checked', false);
			}
		}else{
			$('.option-show').addClass('hide');
		}
	});

	$('#is_public').on('change', function() {
		'use strict';

		var input_name_status = $('input[id="is_public"]').is(":checked");

		if(input_name_status == true){
			var specific_staff = $('input[id="specific_staff"]').is(":checked");
			if(specific_staff == true){
      			$('input[id="specific_staff"]').prop('checked', false);
			}
		}
		$('.option-show').addClass('hide');
	});

</script>