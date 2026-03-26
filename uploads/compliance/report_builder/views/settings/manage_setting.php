<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>
<?php init_head(); ?>

<div id="wrapper">
	<div class="content">
		<div class="row">

			<div class="col-md-3">
				<ul class="nav navbar-pills navbar-pills-flat nav-tabs nav-stacked">
					<?php
					$i = 0;
					foreach($tab as $gr){
						?>
						<li<?php if($i == 0){echo " class='active'"; } ?>>
						<a href="<?php echo admin_url('report_builder/setting?group='.$gr); ?>" data-group="<?php echo new_html_entity_decode($gr); ?>">
							<?php
							$icon['general_setting'] = '<span class="fa fa-area-chart"></span>';
							$icon['category'] = '<span class="fa fa-certificate"></span>';
							echo new_html_entity_decode($icon[$gr] .' '. _l($gr)); 
							?>
						</a>
					</li>
					<?php $i++; } ?>
				</ul>
			</div>
			<div class="col-md-9">
				<div class="panel_s">
					<div class="panel-body">

						<?php $this->load->view($tabs['view']); ?>

					</div>
				</div>
			</div>
			<div class="clearfix"></div>
		</div>
		<?php echo form_close(); ?>
		<div class="btn-bottom-pusher"></div>
	</div>
</div>
<div id="new_version"></div>
<?php init_tail(); ?>

<?php 
$viewuri = $_SERVER['REQUEST_URI'];
?>

<?php if(!(strpos($viewuri,'admin/report_builder/setting?group=general_setting') === false)){ 
	require 'modules/report_builder/assets/js/settings/general_setting_js.php';
}
require 'modules/report_builder/assets/js/settings/category_js.php';

?>
</body>
</html>
