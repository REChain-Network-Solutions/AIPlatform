# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Refund\RefundCancellationDetailsReasonCode
### Namespace: [\YooKassa\Model\Refund](../namespaces/yookassa-model-refund.md)
---
**Summary:**

Класс, представляющий модель RefundCancellationDetailsReasonCode.

**Description:**

Возможные причины отмены возврата:
- `general_decline` - Причина не детализирована
- `insufficient_funds` - Не хватает денег, чтобы сделать возврат
- `rejected_by_payee` - Эмитент платежного средства отклонил возврат по неизвестным причинам
- `yoo_money_account_closed` - Пользователь закрыл кошелек ЮMoney, на который вы пытаетесь вернуть платеж
- `payment_basket_id_not_found` - НСПК не нашла для этого возврата одобренную корзину покупки
- `payment_article_number_not_found` - Указаны товары, для оплаты которых не использовался электронный сертификат: значение `payment_article_number` отсутствует
- `payment_tru_code_not_found` - Указаны товары, для оплаты которых не использовался электронный сертификат: значение `tru_code` отсутствует
- `too_many_refunding_articles` - Для одного или нескольких товаров количество возвращаемых единиц (`quantity`) больше, чем указано в одобренной корзине покупки
- `some_articles_already_refunded` - Некоторые товары уже возвращены
- `rejected_by_timeout` - Технические неполадки на стороне инициатора отмены возврата.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [GENERAL_DECLINE](../classes/YooKassa-Model-Refund-RefundCancellationDetailsReasonCode.md#constant_GENERAL_DECLINE) |  | Причина не детализирована. Для уточнения подробностей обратитесь в техническую поддержку. |
| public | [INSUFFICIENT_FUNDS](../classes/YooKassa-Model-Refund-RefundCancellationDetailsReasonCode.md#constant_INSUFFICIENT_FUNDS) |  | Не хватает денег, чтобы сделать возврат: сумма платежей, которые вы получили в день возврата, меньше, чем сам возврат, или есть задолженность. [Что делать в этом случае](https://yookassa.ru/docs/support/payments/refunding#refunding__block) |
| public | [REJECTED_BY_PAYEE](../classes/YooKassa-Model-Refund-RefundCancellationDetailsReasonCode.md#constant_REJECTED_BY_PAYEE) |  | Эмитент платежного средства отклонил возврат по неизвестным причинам. Предложите пользователю обратиться к эмитенту для уточнения подробностей или договоритесь с пользователем о том, чтобы вернуть ему деньги напрямую, не через ЮKassa. |
| public | [YOO_MONEY_ACCOUNT_CLOSED](../classes/YooKassa-Model-Refund-RefundCancellationDetailsReasonCode.md#constant_YOO_MONEY_ACCOUNT_CLOSED) |  | Пользователь закрыл кошелек ЮMoney, на который вы пытаетесь вернуть платеж. Сделать возврат через ЮKassa нельзя. Договоритесь с пользователем напрямую, каким способом вы вернете ему деньги. |
| public | [PAYMENT_BASKET_ID_NOT_FOUND](../classes/YooKassa-Model-Refund-RefundCancellationDetailsReasonCode.md#constant_PAYMENT_BASKET_ID_NOT_FOUND) |  | НСПК не нашла для этого возврата одобренную корзину покупки. Откорректируйте данные и отправьте запрос еще раз с новым ключом идемпотентности. |
| public | [PAYMENT_ARTICLE_NUMBER_NOT_FOUND](../classes/YooKassa-Model-Refund-RefundCancellationDetailsReasonCode.md#constant_PAYMENT_ARTICLE_NUMBER_NOT_FOUND) |  | Указаны товары, для оплаты которых не использовался электронный сертификат: значение `payment_article_number` отсутствует в одобренной корзине покупки. Откорректируйте данные и отправьте запрос еще раз с новым ключом идемпотентности. |
| public | [PAYMENT_TRU_CODE_NOT_FOUND](../classes/YooKassa-Model-Refund-RefundCancellationDetailsReasonCode.md#constant_PAYMENT_TRU_CODE_NOT_FOUND) |  | Указаны товары, для оплаты которых не использовался электронный сертификат: значение tru_code отсутствует в одобренной корзине покупки. Откорректируйте данные и отправьте запрос еще раз с новым ключом идемпотентности. |
| public | [TOO_MANY_REFUNDING_ARTICLES](../classes/YooKassa-Model-Refund-RefundCancellationDetailsReasonCode.md#constant_TOO_MANY_REFUNDING_ARTICLES) |  | Для одного или нескольких товаров количество возвращаемых единиц (`quantity`) больше, чем указано в одобренной корзине покупки. Откорректируйте данные и отправьте запрос еще раз с новым ключом идемпотентности. |
| public | [SOME_ARTICLES_ALREADY_REFUNDED](../classes/YooKassa-Model-Refund-RefundCancellationDetailsReasonCode.md#constant_SOME_ARTICLES_ALREADY_REFUNDED) |  | Некоторые товары уже возвращены. Откорректируйте данные и отправьте запрос еще раз с новым ключом идемпотентности. |
| public | [REJECTED_BY_TIMEOUT](../classes/YooKassa-Model-Refund-RefundCancellationDetailsReasonCode.md#constant_REJECTED_BY_TIMEOUT) |  | Технические неполадки на стороне инициатора отмены возврата. Повторите запрос с новым ключом идемпотентности. |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Refund-RefundCancellationDetailsReasonCode.md#property_validValues) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Refund/RefundCancellationDetailsReasonCode.php](../../lib/Model/Refund/RefundCancellationDetailsReasonCode.php)
* Package: Default
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Refund\RefundCancellationDetailsReasonCode

---
## Constants
<a name="constant_GENERAL_DECLINE" class="anchor"></a>
###### GENERAL_DECLINE
Причина не детализирована. Для уточнения подробностей обратитесь в техническую поддержку.

```php
GENERAL_DECLINE = 'general_decline'
```


<a name="constant_INSUFFICIENT_FUNDS" class="anchor"></a>
###### INSUFFICIENT_FUNDS
Не хватает денег, чтобы сделать возврат: сумма платежей, которые вы получили в день возврата, меньше, чем сам возврат, или есть задолженность. [Что делать в этом случае](https://yookassa.ru/docs/support/payments/refunding#refunding__block)

```php
INSUFFICIENT_FUNDS = 'insufficient_funds'
```


<a name="constant_REJECTED_BY_PAYEE" class="anchor"></a>
###### REJECTED_BY_PAYEE
Эмитент платежного средства отклонил возврат по неизвестным причинам. Предложите пользователю обратиться к эмитенту для уточнения подробностей или договоритесь с пользователем о том, чтобы вернуть ему деньги напрямую, не через ЮKassa.

```php
REJECTED_BY_PAYEE = 'rejected_by_payee'
```


<a name="constant_YOO_MONEY_ACCOUNT_CLOSED" class="anchor"></a>
###### YOO_MONEY_ACCOUNT_CLOSED
Пользователь закрыл кошелек ЮMoney, на который вы пытаетесь вернуть платеж. Сделать возврат через ЮKassa нельзя. Договоритесь с пользователем напрямую, каким способом вы вернете ему деньги.

```php
YOO_MONEY_ACCOUNT_CLOSED = 'yoo_money_account_closed'
```


<a name="constant_PAYMENT_BASKET_ID_NOT_FOUND" class="anchor"></a>
###### PAYMENT_BASKET_ID_NOT_FOUND
НСПК не нашла для этого возврата одобренную корзину покупки. Откорректируйте данные и отправьте запрос еще раз с новым ключом идемпотентности.

```php
PAYMENT_BASKET_ID_NOT_FOUND = 'payment_basket_id_not_found'
```


<a name="constant_PAYMENT_ARTICLE_NUMBER_NOT_FOUND" class="anchor"></a>
###### PAYMENT_ARTICLE_NUMBER_NOT_FOUND
Указаны товары, для оплаты которых не использовался электронный сертификат: значение `payment_article_number` отсутствует в одобренной корзине покупки. Откорректируйте данные и отправьте запрос еще раз с новым ключом идемпотентности.

```php
PAYMENT_ARTICLE_NUMBER_NOT_FOUND = 'payment_article_number_not_found'
```


<a name="constant_PAYMENT_TRU_CODE_NOT_FOUND" class="anchor"></a>
###### PAYMENT_TRU_CODE_NOT_FOUND
Указаны товары, для оплаты которых не использовался электронный сертификат: значение tru_code отсутствует в одобренной корзине покупки. Откорректируйте данные и отправьте запрос еще раз с новым ключом идемпотентности.

```php
PAYMENT_TRU_CODE_NOT_FOUND = 'payment_tru_code_not_found'
```


<a name="constant_TOO_MANY_REFUNDING_ARTICLES" class="anchor"></a>
###### TOO_MANY_REFUNDING_ARTICLES
Для одного или нескольких товаров количество возвращаемых единиц (`quantity`) больше, чем указано в одобренной корзине покупки. Откорректируйте данные и отправьте запрос еще раз с новым ключом идемпотентности.

```php
TOO_MANY_REFUNDING_ARTICLES = 'too_many_refunding_articles'
```


<a name="constant_SOME_ARTICLES_ALREADY_REFUNDED" class="anchor"></a>
###### SOME_ARTICLES_ALREADY_REFUNDED
Некоторые товары уже возвращены. Откорректируйте данные и отправьте запрос еще раз с новым ключом идемпотентности.

```php
SOME_ARTICLES_ALREADY_REFUNDED = 'some_articles_already_refunded'
```


<a name="constant_REJECTED_BY_TIMEOUT" class="anchor"></a>
###### REJECTED_BY_TIMEOUT
Технические неполадки на стороне инициатора отмены возврата. Повторите запрос с новым ключом идемпотентности.

```php
REJECTED_BY_TIMEOUT = 'rejected_by_timeout'
```



---
## Properties
<a name="property_validValues"></a>
#### protected $validValues : array
---
**Type:** <a href="../array"><abbr title="array">array</abbr></a>
Массив принимаемых enum&#039;ом значений
**Details:**



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