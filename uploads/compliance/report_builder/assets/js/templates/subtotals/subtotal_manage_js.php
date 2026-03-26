<script>
	(function($) {
		"use strict";
		$('#lstview').multiselect({
			search: {
				left: '<input type="text" name="q" class="form-control" autocomplete="off" placeholder="Search..." />',
				right: '<input type="text" name="q" class="form-control" autocomplete="off" placeholder="Search..." />',
			}
		});
	})(jQuery); 
	
	$('select[id="lstview_to"]').on('change', function() {
		"use strict";
		var lstview_to = $("select[id='lstview_to']").val();
	});

	$('.data_source_next').on('click', function() {
		"use strict";

		var template_id = $("input[name='report_template_id']").val();
		var type = 'subtotal';
		$.get(admin_url + 'report_builder/get_next_step_report/' + template_id+'/'+type, function (response) {

			if(response.status == true || response.status == 'true'){
				window.location.assign(response.next_link);
			}else{
				alert_float('danger', '<?php echo _l('please_add_relationship_between_table'); ?>');
			}

		}, 'json');

	});


</script>