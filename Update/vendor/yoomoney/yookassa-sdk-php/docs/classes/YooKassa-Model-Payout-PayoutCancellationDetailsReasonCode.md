# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payout\PayoutCancellationDetailsReasonCode
### Namespace: [\YooKassa\Model\Payout](../namespaces/yookassa-model-payout.md)
---
**Summary:**

Класс, представляющий модель PayoutCancellationDetailsReasonCode.

**Description:**

Возможные причины отмены выплаты. Возможные значения:
- `insufficient_funds` - На балансе выплат не хватает денег для проведения выплаты
- `fraud_suspected` - Выплата заблокирована из-за подозрения в мошенничестве
- `one_time_limit_exceeded` - Превышен лимит на разовое зачисление
- `periodic_limit_exceeded` - Превышен лимит выплат за период времени
- `rejected_by_payee` - Эмитент отклонил выплату по неизвестным причинам
- `general_decline` - Причина не детализирована
- `issuer_unavailable` - Эквайер недоступен
- `recipient_not_found` - Для выплат через СБП: получатель не найден
- `recipient_check_failed` - Только для выплат с проверкой получателя. Получатель выплаты не прошел проверку
- `identification_required` - Кошелек ЮMoney не идентифицирован. Пополнение анонимного кошелька запрещено
- `self_employed_annual_limit_exceeded` - Превышен лимит на годовой доход самозанятого.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [INSUFFICIENT_FUNDS](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#constant_INSUFFICIENT_FUNDS) |  | На балансе выплат не хватает денег для проведения выплаты. [Пополните баланс](https://yookassa.ru/docs/support/payouts#balance) и повторите запрос с новым ключом идемпотентности. |
| public | [FRAUD_SUSPECTED](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#constant_FRAUD_SUSPECTED) |  | Выплата заблокирована из-за подозрения в мошенничестве. Следует обратиться к [инициатору отмены выплаты](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#declined-payouts-cancellation-details-party) за уточнением подробностей или выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту) |
| public | [ONE_TIME_LIMIT_EXCEEDED](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#constant_ONE_TIME_LIMIT_EXCEEDED) |  | Превышен [лимит на разовое зачисление](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#specifics). Можно уменьшить размер выплаты, разбить сумму и сделать несколько выплат, выбрать другой способ получения выплат или другое платежное средство (например, другую банковскую карту) |
| public | [PERIODIC_LIMIT_EXCEEDED](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#constant_PERIODIC_LIMIT_EXCEEDED) |  | Превышен [лимит выплат за период времени](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#specifics) (сутки, месяц). Следует выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту). |
| public | [REJECTED_BY_PAYEE](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#constant_REJECTED_BY_PAYEE) |  | Эмитент отклонил выплату по неизвестным причинам. Пользователю следует обратиться к эмитенту за уточнением подробностей или выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту). |
| public | [GENERAL_DECLINE](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#constant_GENERAL_DECLINE) |  | Причина не детализирована. Пользователю следует обратиться к [инициатору отмены выплаты](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#declined-payouts-cancellation-details-party) за уточнением подробностей |
| public | [ISSUER_UNAVAILABLE](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#constant_ISSUER_UNAVAILABLE) |  | Эквайер недоступен. Следует выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту) или повторить запрос позже с новым ключом идемпотентности. |
| public | [RECIPIENT_NOT_FOUND](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#constant_RECIPIENT_NOT_FOUND) |  | Для [выплат через СБП](https://yookassa.ru/developers/payouts/making-payouts/sbp): получатель не найден — в выбранном банке или платежном сервисе не найден счет, к которому привязан указанный номер телефона. Следует повторить запрос с новыми данными и новым ключом идемпотентности или выбрать другой способ получения выплаты. |
| public | [RECIPIENT_CHECK_FAILED](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#constant_RECIPIENT_CHECK_FAILED) |  | Только для выплат с [проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check). Получатель выплаты не прошел проверку: имя получателя не совпало с именем владельца счета, на который необходимо перевести деньги. [Что делать в этом случае](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check#process-results-canceled-recipient-check-failed) |
| public | [IDENTIFICATION_REQUIRED](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#constant_IDENTIFICATION_REQUIRED) |  | Кошелек ЮMoney не идентифицирован. Пополнение анонимного кошелька запрещено. Пользователю необходимо [идентифицировать кошелек](https://yoomoney.ru/page?id=536136) |
| public | [SELF_EMPLOYED_ANNUAL_LIMIT_EXCEEDED](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#constant_SELF_EMPLOYED_ANNUAL_LIMIT_EXCEEDED) |  | Только для [выплат самозанятым](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Превышен лимит на годовой доход самозанятого. Порекомендуйте получателю выплаты проверить свой доход. |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Payout-PayoutCancellationDetailsReasonCode.md#property_validValues) |  | Возвращает список доступных значений |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Payout/PayoutCancellationDetailsReasonCode.php](../../lib/Model/Payout/PayoutCancellationDetailsReasonCode.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Payout\PayoutCancellationDetailsReasonCode

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_INSUFFICIENT_FUNDS" class="anchor"></a>
###### INSUFFICIENT_FUNDS
На балансе выплат не хватает денег для проведения выплаты. [Пополните баланс](https://yookassa.ru/docs/support/payouts#balance) и повторите запрос с новым ключом идемпотентности.

```php
INSUFFICIENT_FUNDS = 'insufficient_funds'
```


<a name="constant_FRAUD_SUSPECTED" class="anchor"></a>
###### FRAUD_SUSPECTED
Выплата заблокирована из-за подозрения в мошенничестве. Следует обратиться к [инициатору отмены выплаты](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#declined-payouts-cancellation-details-party) за уточнением подробностей или выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту)

```php
FRAUD_SUSPECTED = 'fraud_suspected'
```


<a name="constant_ONE_TIME_LIMIT_EXCEEDED" class="anchor"></a>
###### ONE_TIME_LIMIT_EXCEEDED
Превышен [лимит на разовое зачисление](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#specifics). Можно уменьшить размер выплаты, разбить сумму и сделать несколько выплат, выбрать другой способ получения выплат или другое платежное средство (например, другую банковскую карту)

```php
ONE_TIME_LIMIT_EXCEEDED = 'one_time_limit_exceeded'
```


<a name="constant_PERIODIC_LIMIT_EXCEEDED" class="anchor"></a>
###### PERIODIC_LIMIT_EXCEEDED
Превышен [лимит выплат за период времени](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#specifics) (сутки, месяц). Следует выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту).

```php
PERIODIC_LIMIT_EXCEEDED = 'periodic_limit_exceeded'
```


<a name="constant_REJECTED_BY_PAYEE" class="anchor"></a>
###### REJECTED_BY_PAYEE
Эмитент отклонил выплату по неизвестным причинам. Пользователю следует обратиться к эмитенту за уточнением подробностей или выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту).

```php
REJECTED_BY_PAYEE = 'rejected_by_payee'
```


<a name="constant_GENERAL_DECLINE" class="anchor"></a>
###### GENERAL_DECLINE
Причина не детализирована. Пользователю следует обратиться к [инициатору отмены выплаты](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts#declined-payouts-cancellation-details-party) за уточнением подробностей

```php
GENERAL_DECLINE = 'general_decline'
```


<a name="constant_ISSUER_UNAVAILABLE" class="anchor"></a>
###### ISSUER_UNAVAILABLE
Эквайер недоступен. Следует выбрать другой способ получения выплаты или другое платежное средство (например, другую банковскую карту) или повторить запрос позже с новым ключом идемпотентности.

```php
ISSUER_UNAVAILABLE = 'issuer_unavailable'
```


<a name="constant_RECIPIENT_NOT_FOUND" class="anchor"></a>
###### RECIPIENT_NOT_FOUND
Для [выплат через СБП](https://yookassa.ru/developers/payouts/making-payouts/sbp): получатель не найден — в выбранном банке или платежном сервисе не найден счет, к которому привязан указанный номер телефона. Следует повторить запрос с новыми данными и новым ключом идемпотентности или выбрать другой способ получения выплаты.

```php
RECIPIENT_NOT_FOUND = 'recipient_not_found'
```


<a name="constant_RECIPIENT_CHECK_FAILED" class="anchor"></a>
###### RECIPIENT_CHECK_FAILED
Только для выплат с [проверкой получателя](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check). Получатель выплаты не прошел проверку: имя получателя не совпало с именем владельца счета, на который необходимо перевести деньги. [Что делать в этом случае](https://yookassa.ru/developers/payouts/scenario-extensions/recipient-check#process-results-canceled-recipient-check-failed)

```php
RECIPIENT_CHECK_FAILED = 'recipient_check_failed'
```


<a name="constant_IDENTIFICATION_REQUIRED" class="anchor"></a>
###### IDENTIFICATION_REQUIRED
Кошелек ЮMoney не идентифицирован. Пополнение анонимного кошелька запрещено. Пользователю необходимо [идентифицировать кошелек](https://yoomoney.ru/page?id=536136)

```php
IDENTIFICATION_REQUIRED = 'identification_required'
```


<a name="constant_SELF_EMPLOYED_ANNUAL_LIMIT_EXCEEDED" class="anchor"></a>
###### SELF_EMPLOYED_ANNUAL_LIMIT_EXCEEDED
Только для [выплат самозанятым](https://yookassa.ru/developers/payouts/scenario-extensions/self-employed). Превышен лимит на годовой доход самозанятого. Порекомендуйте получателю выплаты проверить свой доход.

```php
SELF_EMPLOYED_ANNUAL_LIMIT_EXCEEDED = 'self_employed_annual_limit_exceeded'
```



---
## Properties
<a name="property_validValues"></a>
#### protected $validValues : array
---
**Summary**

Возвращает список доступных значений

**Type:** <a href="../array"><abbr title="array">array</abbr></a>
Массив принимаемых enum&#039;ом значений
**Details:**


##### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| return |  |  |


---
## Methods
<a name="method_getEnabledValues" class="anchor"></a>
#### public getEnabledValues() : string[]

```php
Static public getEnabledValues() : string[]
```

**Summary**

Возвращает значения в enum'е значения которых разрешены.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

**Returns:** string[] - Массив разрешённых значений


<a name="method_getValidValues" class="anchor"></a>
#### public getValidValues() : array

```php
Static public getValidValues() : array
```

**Summary**

Возвращает все значения в enum'e.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

**Returns:** array - Массив значений в перечислении


<a name="method_valueExists" class="anchor"></a>
#### public valueExists() : bool

```php
Static public valueExists(mixed $value) : bool
```

**Summary**

Проверяет наличие значения в enum'e.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | value  | Проверяемое значение |

**Returns:** bool - True если значение имеется, false если нет



---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney