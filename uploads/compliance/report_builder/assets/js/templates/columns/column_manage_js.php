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
</script>