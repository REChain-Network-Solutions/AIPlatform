# Multi-Agent Human Inbox — Usage & QA Guide

Team members can now work the External Chatbot's **Human Agent** inbox alongside the account owner, with per-member access control and conversation assignment.

---

## 1. What changed

Previously, only the account that owns a chatbot could see and answer its conversations in the Human Agent inbox. Now:

- The owner can grant any **active team member** access to the inbox with a per-member toggle.
- Granted members see the owner's chatbots and conversations, reply in real time, and their name appears on each message they send.
- Conversations can be **assigned**: agents claim conversations for themselves; the owner can assign anyone on the team. A new filter shows *Mine / Unassigned / All*.
- Security hardening shipped alongside: every conversation action (reply, close, pin, rename, delete, read, assign) is now authorized per conversation. Users outside the team receive `403` when targeting a conversation by ID.

Chatbot **management** is unchanged — creating, editing, training, and deleting chatbots, analytics, and channel settings remain owner-only.

## 2. Prerequisites

- **Extensions:** `chatbot` (External Chatbot) and `chatbot-agent` (Human Agent) installed and registered.
- **Team feature enabled:** admin switch **Admin → Settings → General → "Team Functionality"** must be on (also available under Admin → Config → Tools), and the owner's plan must be a team plan (`is_team_plan`) with available seats.
- **Plan flag:** the owner's plan must have *Enable Human Agent* (`chatbot_human_agent`) turned on. Team members inherit this from the owner's plan — they do not need their own subscription.
- **Realtime:** Ably public/private keys configured (Dashboard → Admin → Settings → Ably) for live message delivery. Without keys the inbox still works but only updates on refresh/fetch.

## 3. Owner setup

1. Go to the **Team** item in the user sidebar (direct URL: `/dashboard/user/team`) and invite the member by email. Member registers through the invite; their status becomes *Active*.

   > **Can't see the Team menu item?** It only appears when all three are true: the admin switch *Team Functionality* is on (Admin → Settings → General), at least one plan with *Team Plan* enabled exists, and the logged-in account is not itself a team member. The direct URL returns 404 while the admin switch is off.
2. Open **Team → member → Edit**. A new switch appears: **"Human Agent inbox access."** Turn it on and save. (The switch only renders when the Chatbot Agent extension is installed.)
3. Done. The member now sees the **Human Agent** menu item and the owner's conversations in it.

> **Access rule:** a member gets inbox access only when all three are true: they belong to the owner's team, their member status is *Active*, and the *Human Agent inbox access* toggle is on. Changing any of these takes effect immediately on the member's next request.

## 4. Working as an agent (team member)

- **Inbox scope:** the member sees the owner's chatbots in the chatbot filter and all their conversations. If the member also owns chatbots of their own, both sets appear merged.
- **Replying:** identical to the owner — replies, internal notes, file attachments, emoji, message rewrite. Replies reach the customer widget (and WhatsApp/Telegram channels) exactly as owner replies do.
- **Attribution:** each human message bubble now shows the *sender's name* instead of the generic "Human Agent" label, so multiple agents in one conversation are distinguishable.
- **Canned responses:** the inbox dropdown shows the **owner's** canned responses (read-only for members). Members do not manage the owner's set.
- **Realtime:** incoming customer messages appear live for the owner and all granted members simultaneously. Unread badge counts include the owner's conversations.

## 5. Assignment

New controls in the conversation header (desktop):

- **Badge:** when a conversation is assigned, a pill shows the assignee's name.
- **Member controls:** a *Claim* button (person-plus icon) when unassigned; a *Release* button (person-minus icon) when assigned to themselves. Members cannot assign anyone else.
- **Owner control:** an *Assign Agent* dropdown listing the owner ("Owner") and every active member with the toggle on, plus *Unassign*.
- **Filter:** new dropdown in the conversation list — *All / Mine / Unassigned*. "Mine" shows conversations assigned to the logged-in user.

> **Rules:** assigning does not lock a conversation — any agent with access can still reply. Concurrent claims resolve last-write-wins. Assigning a user who is not the owner or an eligible agent returns a validation error (`422`).

## 6. Permission matrix

| Action | Owner | Granted member | Ungranted / cancelled member | Outside user |
|---|---|---|---|---|
| See owner's conversations | Yes | Yes | No (empty list) | No (empty list) |
| Reply / internal note | Yes | Yes | 403 | 403 |
| Close / pin / rename / delete conversation | Yes | Yes | 403 | 403 |
| Claim / release self | Yes | Yes | 403 | 403 |
| Assign another agent | Yes | 403 | 403 | 403 |
| See owner's canned responses | Yes | Yes | Own set only | Own set only |
| Manage chatbots / analytics / channels | Yes | No | No | No |
| Toggle member inbox access | Yes | 404 | 404 | 404 |

## 7. Access revocation

- **Toggle off** or **status → Cancelled:** access ends immediately; existing assignments stay on the conversation until the owner reassigns.
- **Member removed from team:** access ends immediately and all conversations assigned to that member on the owner's chatbots are automatically unassigned.
- **Owner plan downgrade** (Human Agent off): the inbox returns `403` for the owner and all members alike.
- **Member account deleted:** their assignments become unassigned (database-level).

## 8. Known limitations (by design, this release)

- Assignment changes are **not pushed in realtime**; other agents see them after their next list fetch or refresh.
- Assignment does not restrict replying — it is a routing/ownership marker, not a lock.
- Realtime uses the Ably public key without token auth (pre-existing architecture; unchanged).
- Members cannot create or edit canned responses for the team.

## 9. QA checklist

Automated coverage exists in `tests/Feature/ChatbotAgentTeamAccessTest.php` (15 tests). The list below is the manual browser pass.

| # | Scenario | Expected |
|---|---|---|
| QA-01 | Owner invites member, member registers, owner enables the toggle on the member edit page. | Toggle persists after save; member status Active. |
| QA-02 | Member logs in, opens Human Agent. | Owner's chatbots in filter; owner's conversations listed. |
| QA-03 | Customer sends a message from the widget while member has the inbox open. | Message appears live for both owner and member (Ably configured). |
| QA-04 | Member replies to a conversation. | Customer widget receives the reply; bubble label shows the member's name. |
| QA-05 | Member opens canned responses dropdown. | Owner's canned responses are listed. |
| QA-06 | Member clicks Claim on an unassigned conversation. | Badge shows member's name; conversation appears under the "Mine" filter. |
| QA-07 | Owner opens Assign Agent dropdown and assigns a member; then Unassign. | Both succeed; badge updates; "Unassigned" filter reflects it. |
| QA-08 | Test Mine / Unassigned / All filter with a mix of assigned and unassigned conversations. | Each filter returns the correct subset. |
| QA-09 | Owner turns the toggle off; member refreshes. | Inbox empties for the member; direct conversation actions return 403. |
| QA-10 | Owner removes the member from the team while member holds assignments. | Member loses access; their assignments become unassigned. |
| QA-11 | Log in as an unrelated account; replay a close/reply/delete request with another tenant's `conversation_id` (e.g. via devtools). | 403 on every mutation; session-history endpoint returns empty. |
| QA-12 | Owner on a plan without Human Agent; member tries the inbox. | 403 for both owner and member. |
| QA-13 | Member who also owns their own chatbot opens the inbox. | Sees both their own and the owner's conversations; realtime works for both. |
| QA-14 | Regression: owner-only pages (chatbot edit, training, analytics) as a member. | Unchanged — member has no access. |

> **Out of scope:** two pre-existing test failures on `staging` are unrelated to this feature: `AIAgentToolChatbotTest` (missing ai-agent tool config) and `ChatbotAnalyticsTest` page-load 500. Both fail on a clean tree without these changes.
