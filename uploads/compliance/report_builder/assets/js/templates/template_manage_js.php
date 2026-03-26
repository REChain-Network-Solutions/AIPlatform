
<script>

	(function($) {
		"use strict";

		var InvoiceServerParams={
			"category_filter"    : "select[name='category_filter[]']",
		};

		var template_table = $('.table-template_table');
		initDataTable(template_table, admin_url+'report_builder/template_table',[0],[0], InvoiceServerParams, [0 ,'desc']);

		$('#category_filter').on('change', function() {
			template_table.DataTable().ajax.reload();
		});
		
		$('#date_add').on('change', function() {
			template_table.DataTable().ajax.reload().columns.adjust().responsive.recalc();
		});

		var hidden_columns = [0];
		$('.table-template_table').DataTable().columns(hidden_columns).visible(false, false);
	})(jQuery); 

</script>