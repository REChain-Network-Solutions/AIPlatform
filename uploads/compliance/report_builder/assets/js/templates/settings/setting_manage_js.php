<script>

	$('.data_source_next').on('click', function() {
		"use strict";

		var template_id = $("input[name='report_template_id']").val();
		var type = 'setting';
		$.get(admin_url + 'report_builder/get_next_step_report/' + template_id+'/'+type, function (response) {

			if(response.status == true || response.status == 'true'){
				window.location.assign(response.next_link);
			}else{
				alert_float('danger', '<?php echo _l('please_add_relationship_between_table'); ?>');
			}

		}, 'json');

	});
	
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
				$('.option-show').addClass('hide');
			}
		}
	});


</script>