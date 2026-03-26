<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>

<?php init_head(); ?>
<div id="wrapper">
	<div class="content">
		<div class="row">
			<div class="col-md-12">
				<div class="panel_s"> 
					<div class="panel-body">

						<div class="row">
							<div class="col-md-12">
								<h4 class="h4-color no-margin"><i class="fa fa-outdent menu-icon" aria-hidden="true"></i> <?php echo _l('report_template'); ?></h4>
							</div>
						</div>
						<hr class="hr-color">

						<div class="row">
							
							<div  class="col-md-3 leads-filter-column">
								<select name="category_filter[]" id="category_filter" data-live-search="true" class="selectpicker" multiple="true" data-actions-box="true" data-width="100%" data-none-selected-text="<?php echo _l('category'); ?>">
									<?php foreach($categories as $category) { ?>
										<option value="<?php echo new_html_entity_decode($category['id']); ?>"><?php echo new_html_entity_decode($category['name']); ?></option>
									<?php } ?>
								</select>
							</div> 

						</div>
						<br>

						<?php render_datatable(array(
							_l('id'),
							_l('report_title'),
							_l('category_name'),
							_l('rb_staff_create'),
							_l('rp_date_create'),
							_l('rb_options'),
						),'template_table'); ?>
					</div>

				</div>
			</div>

		</div>
	</div>
</div>
<?php init_tail(); ?>
<?php 
require('modules/report_builder/assets/js/templates/template_manage_js.php');
?>
</body>
</html>
