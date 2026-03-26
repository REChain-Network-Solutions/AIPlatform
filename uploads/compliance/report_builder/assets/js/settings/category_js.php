<script>
	(function($) {
		"use strict"; 
		appValidateForm($('#category_setting'), {
			name: 'required',
		}); 
	})(jQuery); 
	
	function new_category(){
		"use strict";

		$('#category').modal('show');
		$('.edit-title').addClass('hide');
		$('.add-title').removeClass('hide');
		$('#category_id_t').html('');

		$('#category_setting input[name="name"]').val('');

	}

	function edit_category(invoker,id){
		"use strict";

		$('#category').modal('show');
		$('.edit-title').removeClass('hide');
		$('.add-title').addClass('hide');

		$('#category_id_t').html('');
		$('#category_id_t').append(hidden_input('id',id));

		$('#category_setting input[name="name"]').val($(invoker).data('name'));

	}
</script>