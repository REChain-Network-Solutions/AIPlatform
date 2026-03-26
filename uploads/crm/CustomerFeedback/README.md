# CustomerFeedback Module - Unified Documentation

## Overview

**CustomerFeedback** is a unified, enterprise-grade customer feedback and complaint management system that consolidates NPS surveys, CSAT measurement, complaint tracking, and AI-powered insights into a single cohesive module.

This module is the result of merging two separate modules (Feedback & Complaint) into one unified system with:
- ✓ Single source of truth via `FeedbackTicket` entity
- ✓ Type discrimination using `feedback_type` enum
- ✓ Full backward compatibility with legacy Complaint routes
- ✓ AI-powered insights and sentiment analysis
- ✓ Multi-channel feedback ingestion (email, web form, API, SMS, chat)
- ✓ Comprehensive analytics dashboard
- ✓ NPS & CSAT survey frameworks with automated response tracking
- ✓ Email synchronization with IMAP/POP3 support
- ✓ Agent assignment with group hierarchy
- ✓ Reply templates and automation
- ✓ File attachments and media handling
- ✓ Soft deletes and company multi-tenancy

---

## Installation

### Step 1: Copy Module
Place the `CustomerFeedback` folder in your `Modules` directory:
```bash
cp -r CustomerFeedback /path/to/your/project/Modules/
```

### Step 2: Register Module
Update `modules_statuses.json` to enable the module:
```json
{
    "CustomerFeedback": 1
}
```

### Step 3: Install Dependencies
```bash
composer dump-autoload
```

### Step 4: Run Migrations
```bash
php artisan migrate
```

### Step 5: Publish Assets
```bash
php artisan vendor:publish --tag=customer-feedback-migrations
php artisan vendor:publish --tag=customer-feedback-config
```

---

## Architecture Overview

### Entity Relationship Diagram

```
FeedbackTicket (main entity)
├── FeedbackReply (1:many) - Thread responses
├── FeedbackFile (1:many) - Attachments
├── FeedbackChannel (many:1) - Communication channel
├── FeedbackType (many:1) - Classification type
├── FeedbackGroup (many:1) - Agent group assignment
├── FeedbackTag / FeedbackTagList (many:many) - Tagging
├── NpsResponse (1:1) - NPS survey response
├── CsatResponse (1:1) - CSAT survey response
└── FeedbackInsight (1:many) - AI analysis results

NpsSurvey
├── NpsResponse (1:many)
└── FeedbackTicket (implicit via response)

CsatSurvey
├── CsatResponse (1:many)
└── FeedbackTicket (implicit via response)

FeedbackGroup
└── FeedbackAgentGroups (pivot) → User agents

FeedbackReplyTemplate (canned responses)

FeedbackCustomForm (dynamic form builder)

FeedbackEmailSetting (IMAP configuration)
```

### Database Tables (19 total)

| Table | Purpose | Records |
|-------|---------|---------|
| `feedback_tickets` | Main ticket/feedback entity | All tickets |
| `feedback_replies` | Thread responses | All replies |
| `feedback_files` | Attachments | All uploaded files |
| `feedback_channels` | Communication sources | Channel definitions |
| `feedback_types` | Classification types | Type definitions |
| `feedback_groups` | Agent grouping | Group definitions |
| `feedback_agent_groups` | Agent-to-group mapping | Agent assignments |
| `feedback_tags_list` | Tag catalog | Tag definitions |
| `feedback_tags` | Ticket-to-tag mapping | Tag assignments |
| `feedback_reply_templates` | Canned responses | Response templates |
| `feedback_custom_forms` | Dynamic form definitions | Form definitions |
| `feedback_email_settings` | IMAP configuration | Email config |
| `nps_surveys` | NPS survey definitions | NPS surveys |
| `nps_responses` | Individual NPS scores | Survey responses |
| `csat_surveys` | CSAT survey definitions | CSAT surveys |
| `csat_responses` | Individual CSAT scores | Survey responses |
| `feedback_insights` | AI analysis results | AI insights |

---

## Key Features

### 1. Unified Ticket Management
Create, read, update, delete feedback tickets with type discrimination:
```php
$ticket = FeedbackTicket::create([
    'title' => 'Service Issue',
    'description' => 'Product stopped working',
    'feedback_type' => FeedbackTicket::TYPE_COMPLAINT,
    'status' => FeedbackTicket::STATUS_OPEN,
    'priority' => FeedbackTicket::PRIORITY_HIGH,
]);

// Use scopes for filtering
$complaints = FeedbackTicket::complaints()->unresolved()->get();
$feedback = FeedbackTicket::feedback()->highPriority()->get();
```

### 2. Reply Threading
Complete conversation management with public/internal replies:
```php
$reply = FeedbackReply::create([
    'feedback_id' => $ticket->id,
    'user_id' => auth()->id(),
    'message' => 'Thank you for your feedback...',
    'is_internal' => false, // visible to client
    'source_channel' => FeedbackReply::SOURCE_PORTAL,
]);

// Get public replies only
$publicReplies = $ticket->replies()->public()->get();
```

### 3. NPS Surveys
Net Promoter Score surveys with automatic categorization:
```php
$survey = NpsSurvey::create([
    'title' => 'Product Satisfaction',
    'question' => 'How likely to recommend?',
    'status' => true,
]);

// Submit response
$response = NpsResponse::create([
    'nps_survey_id' => $survey->id,
    'user_id' => auth()->id(),
    'score' => 9,
    'feedback' => 'Great product!',
]);

// Auto-create complaint ticket for detractors (score <= 6)
if ($response->score <= 6) {
    $ticket = FeedbackTicket::create([
        'feedback_type' => FeedbackTicket::TYPE_SURVEY_RESPONSE,
        'status' => FeedbackTicket::STATUS_OPEN,
        'priority' => FeedbackTicket::PRIORITY_HIGH,
    ]);
}
```

### 4. CSAT Surveys
Customer Satisfaction surveys with response tracking:
```php
$survey = CsatSurvey::create([
    'title' => 'Service Satisfaction',
    'question' => 'How satisfied with our service?',
    'scale_min' => 1,
    'scale_max' => 5,
]);
```

### 5. AI-Powered Insights
Automatic sentiment analysis, category suggestion, priority detection:
```php
// Get insights for a ticket
$insights = $ticket->insights()->get();

// Each insight contains:
// - insight_type: sentiment|category|priority|action|trend
// - confidence_score: 0.0-1.0
// - suggested_action: Recommended action
```

### 6. Multi-Channel Ingestion
Support for email, web forms, API, SMS, chat:
```php
$channels = FeedbackChannel::where('status', true)->get();
// email, web_form, api, sms, chat, phone
```

### 7. Email Synchronization
IMAP/POP3 email sync with automatic ticket creation:
```php
$setting = FeedbackEmailSetting::create([
    'imap_host' => 'imap.gmail.com',
    'imap_port' => 993,
    'imap_encryption' => 'ssl',
    'imap_username' => 'feedback@company.com',
    'imap_password' => encrypt('password'),
    'email_address' => 'feedback@company.com',
    'auto_reply' => true,
    'reply_message' => 'Thank you for your email...',
]);
```

### 8. Agent Management
Group-based agent assignment with load balancing:
```php
$group = FeedbackGroup::create([
    'name' => 'Customer Support Team',
    'description' => 'Primary support group',
    'status' => true,
]);

// Add agents to group
$group->agents()->attach($userId, ['added_by' => auth()->id()]);

// Assign ticket to group or specific agent
$ticket->update(['group_id' => $group->id, 'agent_id' => $agentId]);
```

### 9. Reply Templates
Pre-defined responses for common scenarios:
```php
$template = FeedbackReplyTemplate::create([
    'name' => 'Generic Acknowledgment',
    'message' => 'Thank you for your feedback. We will review and respond shortly.',
    'reply_type' => FeedbackReplyTemplate::TYPE_AUTO,
    'status' => true,
]);
```

### 10. Analytics Dashboard
Comprehensive metrics and trend analysis:
```php
// Key metrics
- Total tickets
- Open tickets
- Average resolution time
- Satisfaction score
- Status breakdown
- Priority distribution
- Feedback type breakdown
- Top performing agents
- NPS trend
- CSAT trend
```

---

## API Endpoints

### Tickets
```
GET    /api/feedback/tickets              List all tickets
POST   /api/feedback/tickets              Create ticket
GET    /api/feedback/tickets/{id}         Get single ticket
PUT    /api/feedback/tickets/{id}         Update ticket
DELETE /api/feedback/tickets/{id}         Delete ticket
```

### Replies
```
GET    /api/feedback/tickets/{id}/replies    Get all replies
POST   /api/feedback/tickets/{id}/replies    Add reply
DELETE /api/feedback/replies/{id}            Delete reply
```

### Surveys
```
POST   /api/feedback/surveys/nps          Create NPS survey
GET    /api/feedback/surveys/nps/{id}     Get NPS survey
POST   /api/feedback/surveys/{id}/respond Submit NPS response (public)
```

### Analytics
```
GET    /customer-feedback/analytics/dashboard   Dashboard metrics
GET    /customer-feedback/analytics/nps        NPS analytics
GET    /customer-feedback/analytics/csat       CSAT analytics
```

### Insights
```
GET    /customer-feedback/insights/dashboard                  Insights dashboard
GET    /customer-feedback/insights/tickets/{id}               Get ticket insights
POST   /customer-feedback/insights/tickets/{id}/analyze       Analyze ticket
GET    /customer-feedback/insights/tickets/{id}/sentiment     Get sentiment
GET    /customer-feedback/insights/tickets/{id}/category      Get category suggestion
GET    /customer-feedback/insights/tickets/{id}/priority      Get priority suggestion
GET    /customer-feedback/insights/tickets/{id}/response      Get response suggestion
```

---

## Scopes & Query Helpers

```php
// Filter by type
FeedbackTicket::complaints()->get()
FeedbackTicket::feedback()->get()
FeedbackTicket::surveyResponses()->get()

// Filter by status
FeedbackTicket::unresolved()->get()

// Filter by priority
FeedbackTicket::highPriority()->get()

// Reply scopes
FeedbackReply::public()->get()
FeedbackReply::internal()->get()
FeedbackReply::aiGenerated()->get()

// Survey scopes
FeedbackType::complaints()->get()
FeedbackType::feedback()->get()
FeedbackType::surveys()->get()
```

---

## Events & Listeners

### Events Fired
1. **FeedbackTicketCreated** - When ticket is created
2. **FeedbackTicketUpdated** - When ticket is updated
3. **FeedbackReplyAdded** - When reply is added
4. **NpsSurveyCreated** - When NPS survey is created

### Listeners Registered
1. **SendFeedbackNotification** - Notifies assigned agents
2. **TriggerFeedbackAnalysis** - Queues AI analysis job

---

## Configuration

Edit `/Config/config.php`:

```php
'module_settings' => [
    'enable_nps' => true,
    'enable_csat' => true,
    'enable_ai_insights' => true,
    'enable_email_sync' => true,
],

'ai' => [
    'enabled' => true,
    'sentiment_analysis' => true,
    'category_suggestion' => true,
    'priority_suggestion' => true,
    'response_suggestion' => true,
    'min_confidence' => 0.7,
],

'survey' => [
    'nps' => [
        'default_question' => 'How likely are you to recommend us?',
        'scale_min' => 1,
        'scale_max' => 10,
    ],
    'csat' => [
        'default_question' => 'How satisfied are you?',
        'scale_min' => 1,
        'scale_max' => 5,
    ],
],
```

---

## Migration from Dual Modules

If you have existing Feedback and Complaint modules:

1. **Backup Data**
   ```bash
   php artisan backup:run
   ```

2. **Export Legacy Data**
   ```php
   // Run script to export complaint and feedback data
   ```

3. **Disable Old Modules**
   ```json
   {
       "Feedback": 0,
       "Complaint": 0,
       "CustomerFeedback": 1
   }
   ```

4. **Run Migrations**
   ```bash
   php artisan migrate
   ```

5. **Import Data**
   ```php
   // Run script to import exported data into new structure
   ```

6. **Update Routes**
   Legacy routes (e.g., `/complaint`) are automatically mapped to `/customer-feedback`

---

## Usage Examples

### Create a Complaint Ticket
```php
$ticket = FeedbackTicket::create([
    'company_id' => company()->id,
    'user_id' => auth()->id(),
    'title' => 'Order not delivered',
    'description' => 'I placed an order 5 days ago and it hasn\'t arrived',
    'feedback_type' => FeedbackTicket::TYPE_COMPLAINT,
    'status' => FeedbackTicket::STATUS_OPEN,
    'priority' => FeedbackTicket::PRIORITY_HIGH,
    'channel_id' => 1, // Email channel
]);
```

### Add a Reply
```php
$ticket->replies()->create([
    'user_id' => auth()->id(),
    'message' => 'We apologize for the delay. Let us investigate.',
    'is_internal' => false,
]);
```

### Create NPS Survey
```php
$survey = NpsSurvey::create([
    'company_id' => company()->id,
    'title' => 'Q1 2025 Satisfaction',
    'question' => 'How likely to recommend?',
]);

// Get results
$total = $survey->responses()->count();
$promoters = $survey->responses()->where('score', '>=', 9)->count();
$detractors = $survey->responses()->where('score', '<=', 6)->count();
$npsScore = (($promoters - $detractors) / $total) * 100;
```

### View Analytics
```php
// Get dashboard data
$metrics = [
    'totalTickets' => FeedbackTicket::count(),
    'openTickets' => FeedbackTicket::unresolved()->count(),
    'avgResolutionTime' => // calculated from resolved tickets
    'satisfactionScore' => // NPS + CSAT average
];
```

---

## Testing

Run the test suite:
```bash
php artisan test --filter=CustomerFeedback
```

Tests are located in `/Tests/Feature` and `/Tests/Unit`.

---

## Troubleshooting

### Email Sync Issues
- Check IMAP credentials in `feedback_email_settings`
- Ensure SSL/TLS port is correct (993 for SSL, 143 for TLS)
- Verify firewall allows IMAP connections

### AI Insights Not Working
- Ensure `ai.enabled` is true in config
- Check minimum confidence score threshold
- Verify AI service is configured

### Permission Errors
- Check user has `view_feedback` permission
- Verify company_id matches
- Check agent group assignments

---

## Support

For issues or questions:
1. Check the module documentation
2. Review error logs in `storage/logs/`
3. Check database for data consistency
4. Review permission settings

---

## License

This module is part of TitanZero and follows the same licensing terms.

**Version:** 1.0.0  
**Last Updated:** March 22, 2026  
**Status:** Production Ready
