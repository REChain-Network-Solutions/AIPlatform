<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>
<div>
<div class="_buttons">
    
    <?php if (has_permission('report_builder', '', 'create') || is_admin()) { ?>
    <a href="#" onclick="new_category(); return false;" class="btn btn-info pull-left display-block">
        <?php echo _l('add_category'); ?>
    </a>
<?php } ?>

</div>
<div class="clearfix"></div>
<hr class="hr-panel-heading" />
<div class="clearfix"></div>
<table class="table dt-table border table-striped">
 <thead>
    <th><?php echo _l('rp_order'); ?></th>
    <th><?php echo _l('category_name'); ?></th>
    
    <th><?php echo _l('options'); ?></th>
 </thead>
  <tbody>
    <?php foreach($categories as $category){ ?>

    <tr>
        <td><?php echo new_html_entity_decode($category['id']); ?></td>
        <td><?php echo new_html_entity_decode($category['name']); ?></td>
        <td>
            <?php if (has_permission('report_builder', '', 'edit') || is_admin()) { ?>
              <a href="#" onclick="edit_category(this,<?php echo new_html_entity_decode($category['id']); ?>); return false;" data-name="<?php echo new_html_entity_decode($category['name']); ?>" class="btn btn-default btn-icon"><i class="fa-regular fa-pen-to-square"></i>
            </a>
            <?php } ?>

            <?php if (has_permission('report_builder', '', 'delete') || is_admin()) { ?> 
            <a href="<?php echo admin_url('report_builder/delete_category/'.$category['id']); ?>" class="btn btn-danger btn-icon _delete"><i class="fa fa-remove"></i></a>
             <?php } ?>
        </td>
    </tr>
    <?php } ?>
 </tbody>
</table>   

<div class="modal fade" id="category" tabindex="-1" role="dialog">
        <div class="modal-dialog w-25">
          <?php echo form_open_multipart(admin_url('report_builder/category_setting'), array('id'=>'category_setting')); ?>

            <div class="modal-content">
                <div class="modal-header">
                    <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
                    <h4 class="modal-title">
                        <span class="add-title"><?php echo _l('add_category'); ?></span>
                        <span class="edit-title"><?php echo _l('edit_category'); ?></span>
                    </h4>
                </div>

                <div class="modal-body">
                    <div class="row">
                        <div class="col-md-12">
                             <div id="category_id_t"></div>   
                          <div class="form"> 
                            <div class="col-md-12">
                              <?php echo render_input('name', 'category_name'); ?>
                            </div>
                          </div>
                        </div>
                    </div>
                </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-default" data-dismiss="modal"><?php echo _l('close'); ?></button>
                        
                         <button type="submit" class="btn btn-info intext-btn"><?php echo _l('submit'); ?></button>
                    </div>
                </div>
                <?php echo form_close(); ?>
            </div>
        </div> 
</div>

</body>
</html>
