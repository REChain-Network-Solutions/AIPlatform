# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Invoice\Invoice
### Namespace: [\YooKassa\Model\Invoice](../namespaces/yookassa-model-invoice.md)
---
**Summary:**

Класс, представляющий модель Invoice.

**Description:**

Объект счета (Invoice) — актуальная информация о счете.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$cancellation_details](../classes/YooKassa-Model-Invoice-Invoice.md#property_cancellation_details) |  | Комментарий к статусу `canceled`: кто отменил счет и по какой причине. |
| public | [$cancellationDetails](../classes/YooKassa-Model-Invoice-Invoice.md#property_cancellationDetails) |  | Комментарий к статусу `canceled`: кто отменил счет и по какой причине. |
| public | [$cart](../classes/YooKassa-Model-Invoice-Invoice.md#property_cart) |  | Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой. |
| public | [$created_at](../classes/YooKassa-Model-Invoice-Invoice.md#property_created_at) |  | Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2017-11-03T11:52:31.827Z` |
| public | [$createdAt](../classes/YooKassa-Model-Invoice-Invoice.md#property_createdAt) |  | Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2017-11-03T11:52:31.827Z` |
| public | [$delivery_method](../classes/YooKassa-Model-Invoice-Invoice.md#property_delivery_method) |  | Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`. |
| public | [$deliveryMethod](../classes/YooKassa-Model-Invoice-Invoice.md#property_deliveryMethod) |  | Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`. |
| public | [$description](../classes/YooKassa-Model-Invoice-Invoice.md#property_description) |  | Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37». |
| public | [$expires_at](../classes/YooKassa-Model-Invoice-Invoice.md#property_expires_at) |  | Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2024-10-18T10:51:18.139Z` Присутствует только для счетов в статусе `pending`. |
| public | [$expiresAt](../classes/YooKassa-Model-Invoice-Invoice.md#property_expiresAt) |  | Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2024-10-18T10:51:18.139Z` Присутствует только для счетов в статусе `pending`. |
| public | [$id](../classes/YooKassa-Model-Invoice-Invoice.md#property_id) |  | Идентификатор счета в ЮKassa. |
| public | [$metadata](../classes/YooKassa-Model-Invoice-Invoice.md#property_metadata) |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8. |
| public | [$payment_details](../classes/YooKassa-Model-Invoice-Invoice.md#property_payment_details) |  | Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation). |
| public | [$paymentDetails](../classes/YooKassa-Model-Invoice-Invoice.md#property_paymentDetails) |  | Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation). |
| public | [$status](../classes/YooKassa-Model-Invoice-Invoice.md#property_status) |  | Статус счета. Возможные значения: `pending`, `succeeded`, `canceled`. |
| protected | [$_cancellation_details](../classes/YooKassa-Model-Invoice-Invoice.md#property__cancellation_details) |  | Комментарий к статусу `canceled`: кто отменил счет и по какой причине. |
| protected | [$_cart](../classes/YooKassa-Model-Invoice-Invoice.md#property__cart) |  | Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой. |
| protected | [$_created_at](../classes/YooKassa-Model-Invoice-Invoice.md#property__created_at) |  | Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). |
| protected | [$_delivery_method](../classes/YooKassa-Model-Invoice-Invoice.md#property__delivery_method) |  | Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`. |
| protected | [$_description](../classes/YooKassa-Model-Invoice-Invoice.md#property__description) |  | Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. |
| protected | [$_expires_at](../classes/YooKassa-Model-Invoice-Invoice.md#property__expires_at) |  | Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). |
| protected | [$_id](../classes/YooKassa-Model-Invoice-Invoice.md#property__id) |  | Идентификатор счета в ЮKassa. |
| protected | [$_metadata](../classes/YooKassa-Model-Invoice-Invoice.md#property__metadata) |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). |
| protected | [$_payment_details](../classes/YooKassa-Model-Invoice-Invoice.md#property__payment_details) |  | Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation). |
| protected | [$_status](../classes/YooKassa-Model-Invoice-Invoice.md#property__status) |  | Статус счета. Возможные значения: `pending`, `succeeded`, `canceled`. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getCancellationDetails()](../classes/YooKassa-Model-Invoice-Invoice.md#method_getCancellationDetails) |  | Возвращает cancellation_details. |
| public | [getCart()](../classes/YooKassa-Model-Invoice-Invoice.md#method_getCart) |  | Возвращает cart. |
| public | [getCreatedAt()](../classes/YooKassa-Model-Invoice-Invoice.md#method_getCreatedAt) |  | Возвращает created_at. |
| public | [getDeliveryMethod()](../classes/YooKassa-Model-Invoice-Invoice.md#method_getDeliveryMethod) |  | Возвращает delivery_method. |
| public | [getDescription()](../classes/YooKassa-Model-Invoice-Invoice.md#method_getDescription) |  | Возвращает description. |
| public | [getExpiresAt()](../classes/YooKassa-Model-Invoice-Invoice.md#method_getExpiresAt) |  | Возвращает expires_at. |
| public | [getId()](../classes/YooKassa-Model-Invoice-Invoice.md#method_getId) |  | Возвращает id. |
| public | [getMetadata()](../classes/YooKassa-Model-Invoice-Invoice.md#method_getMetadata) |  | Возвращает metadata. |
| public | [getPaymentDetails()](../classes/YooKassa-Model-Invoice-Invoice.md#method_getPaymentDetails) |  | Возвращает payment_details. |
| public | [getStatus()](../classes/YooKassa-Model-Invoice-Invoice.md#method_getStatus) |  | Возвращает status. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setCancellationDetails()](../classes/YooKassa-Model-Invoice-Invoice.md#method_setCancellationDetails) |  | Устанавливает cancellation_details. |
| public | [setCart()](../classes/YooKassa-Model-Invoice-Invoice.md#method_setCart) |  | Устанавливает cart. |
| public | [setCreatedAt()](../classes/YooKassa-Model-Invoice-Invoice.md#method_setCreatedAt) |  | Устанавливает created_at. |
| public | [setDeliveryMethod()](../classes/YooKassa-Model-Invoice-Invoice.md#method_setDeliveryMethod) |  | Устанавливает delivery_method. |
| public | [setDescription()](../classes/YooKassa-Model-Invoice-Invoice.md#method_setDescription) |  | Устанавливает description. |
| public | [setExpiresAt()](../classes/YooKassa-Model-Invoice-Invoice.md#method_setExpiresAt) |  | Устанавливает expires_at. |
| public | [setId()](../classes/YooKassa-Model-Invoice-Invoice.md#method_setId) |  | Устанавливает id. |
| public | [setMetadata()](../classes/YooKassa-Model-Invoice-Invoice.md#method_setMetadata) |  | Устанавливает metadata. |
| public | [setPaymentDetails()](../classes/YooKassa-Model-Invoice-Invoice.md#method_setPaymentDetails) |  | Устанавливает payment_details. |
| public | [setStatus()](../classes/YooKassa-Model-Invoice-Invoice.md#method_setStatus) |  | Устанавливает status. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Invoice/Invoice.php](../../lib/Model/Invoice/Invoice.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Invoice\Invoice
* Implements:
  * [\YooKassa\Model\Invoice\InvoiceInterface](../classes/YooKassa-Model-Invoice-InvoiceInterface.md)

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Properties
<a name="property_cancellation_details"></a>
#### public $cancellation_details : \YooKassa\Model\Invoice\InvoiceCancellationDetails|null
---
***Description***

Комментарий к статусу `canceled`: кто отменил счет и по какой причине.

**Type:** <a href="../\YooKassa\Model\Invoice\InvoiceCancellationDetails|null"><abbr title="\YooKassa\Model\Invoice\InvoiceCancellationDetails|null">InvoiceCancellationDetails|null</abbr></a>

**Details:**


<a name="property_cancellationDetails"></a>
#### public $cancellationDetails : \YooKassa\Model\Invoice\InvoiceCancellationDetails|null
---
***Description***

Комментарий к статусу `canceled`: кто отменил счет и по какой причине.

**Type:** <a href="../\YooKassa\Model\Invoice\InvoiceCancellationDetails|null"><abbr title="\YooKassa\Model\Invoice\InvoiceCancellationDetails|null">InvoiceCancellationDetails|null</abbr></a>

**Details:**


<a name="property_cart"></a>
#### public $cart : \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface
---
***Description***

Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.

**Type:** <a href="../\YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface"><abbr title="\YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface">ListObjectInterface</abbr></a>

**Details:**


<a name="property_created_at"></a>
#### public $created_at : \DateTime
---
***Description***

Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2017-11-03T11:52:31.827Z`

**Type:** \DateTime

**Details:**


<a name="property_createdAt"></a>
#### public $createdAt : \DateTime
---
***Description***

Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2017-11-03T11:52:31.827Z`

**Type:** \DateTime

**Details:**


<a name="property_delivery_method"></a>
#### public $delivery_method : \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|null
---
***Description***

Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`.

**Type:** <a href="../\YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|null"><abbr title="\YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|null">AbstractDeliveryMethod|null</abbr></a>

**Details:**


<a name="property_deliveryMethod"></a>
#### public $deliveryMethod : \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|null
---
***Description***

Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`.

**Type:** <a href="../\YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|null"><abbr title="\YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|null">AbstractDeliveryMethod|null</abbr></a>

**Details:**


<a name="property_description"></a>
#### public $description : string|null
---
***Description***

Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37».

**Type:** <a href="../string|null"><abbr title="string|null">string|null</abbr></a>

**Details:**


<a name="property_expires_at"></a>
#### public $expires_at : \DateTime|null
---
***Description***

Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2024-10-18T10:51:18.139Z` Присутствует только для счетов в статусе `pending`.

**Type:** <a href="../\DateTime|null"><abbr title="\DateTime|null">DateTime|null</abbr></a>

**Details:**


<a name="property_expiresAt"></a>
#### public $expiresAt : \DateTime|null
---
***Description***

Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: `2024-10-18T10:51:18.139Z` Присутствует только для счетов в статусе `pending`.

**Type:** <a href="../\DateTime|null"><abbr title="\DateTime|null">DateTime|null</abbr></a>

**Details:**


<a name="property_id"></a>
#### public $id : string
---
***Description***

Идентификатор счета в ЮKassa.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : \YooKassa\Model\Metadata|null
---
***Description***

Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.

**Type:** <a href="../\YooKassa\Model\Metadata|null"><abbr title="\YooKassa\Model\Metadata|null">Metadata|null</abbr></a>

**Details:**


<a name="property_payment_details"></a>
#### public $payment_details : \YooKassa\Model\Invoice\PaymentDetails|null
---
***Description***

Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation).

**Type:** <a href="../\YooKassa\Model\Invoice\PaymentDetails|null"><abbr title="\YooKassa\Model\Invoice\PaymentDetails|null">PaymentDetails|null</abbr></a>

**Details:**


<a name="property_paymentDetails"></a>
#### public $paymentDetails : \YooKassa\Model\Invoice\PaymentDetails|null
---
***Description***

Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation).

**Type:** <a href="../\YooKassa\Model\Invoice\PaymentDetails|null"><abbr title="\YooKassa\Model\Invoice\PaymentDetails|null">PaymentDetails|null</abbr></a>

**Details:**


<a name="property_status"></a>
#### public $status : string
---
***Description***

Статус счета. Возможные значения: `pending`, `succeeded`, `canceled`.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property__cancellation_details"></a>
#### protected $_cancellation_details : ?\YooKassa\Model\Invoice\InvoiceCancellationDetails
---
**Summary**

Комментарий к статусу `canceled`: кто отменил счет и по какой причине.

**Type:** <a href="../?\YooKassa\Model\Invoice\InvoiceCancellationDetails"><abbr title="?\YooKassa\Model\Invoice\InvoiceCancellationDetails">InvoiceCancellationDetails</abbr></a>

**Details:**


<a name="property__cart"></a>
#### protected $_cart : ?\YooKassa\Common\ListObjectInterface
---
**Summary**

Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.

**Type:** <a href="../?\YooKassa\Common\ListObjectInterface"><abbr title="?\YooKassa\Common\ListObjectInterface">ListObjectInterface</abbr></a>

**Details:**


<a name="property__created_at"></a>
#### protected $_created_at : ?\DateTime
---
**Summary**

Дата и время создания счета на оплату. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).

***Description***

Пример: ~`2017-11-03T11:52:31.827Z`

**Type:** <a href="../?\DateTime"><abbr title="?\DateTime">DateTime</abbr></a>

**Details:**


<a name="property__delivery_method"></a>
#### protected $_delivery_method : ?\YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod
---
**Summary**

Данные о выбранном способе доставки счета. Присутствует только для счетов в статусе `pending`.

**Type:** <a href="../?\YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod"><abbr title="?\YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod">AbstractDeliveryMethod</abbr></a>

**Details:**


<a name="property__description"></a>
#### protected $_description : ?string
---
**Summary**

Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета.

***Description***

Например: «Счет на оплату по договору 37».

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**


<a name="property__expires_at"></a>
#### protected $_expires_at : ?\DateTime
---
**Summary**

Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601).

***Description***

Пример: `2024-10-18T10:51:18.139Z`

Присутствует только для счетов в статусе `pending`.

**Type:** <a href="../?\DateTime"><abbr title="?\DateTime">DateTime</abbr></a>

**Details:**


<a name="property__id"></a>
#### protected $_id : ?string
---
**Summary**

Идентификатор счета в ЮKassa.

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**


<a name="property__metadata"></a>
#### protected $_metadata : ?\YooKassa\Model\Metadata
---
**Summary**

Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).

***Description***

Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa.
Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.

**Type:** <a href="../?\YooKassa\Model\Metadata"><abbr title="?\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property__payment_details"></a>
#### protected $_payment_details : ?\YooKassa\Model\Invoice\PaymentDetails
---
**Summary**

Данные о платеже по выставленному счету. Присутствуют, только если платеж успешно [подтвержден пользователем](https://yookassa.ru/developers/payment-acceptance/getting-started/payment-process#user-confirmation).

**Type:** <a href="../?\YooKassa\Model\Invoice\PaymentDetails"><abbr title="?\YooKassa\Model\Invoice\PaymentDetails">PaymentDetails</abbr></a>

**Details:**


<a name="property__status"></a>
#### protected $_status : ?string
---
**Summary**

Статус счета. Возможные значения: `pending`, `succeeded`, `canceled`.

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

AbstractObject constructor.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  |  |

**Returns:** mixed - 


<a name="method___get" class="anchor"></a>
#### public __get() : mixed

```php
public __get(string $propertyName) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method___isset" class="anchor"></a>
#### public __isset() : bool

```php
public __isset(string $propertyName) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method___set" class="anchor"></a>
#### public __set() : void

```php
public __set(string $propertyName, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method___unset" class="anchor"></a>
#### public __unset() : void

```php
public __unset(string $propertyName) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getCancellationDetails" class="anchor"></a>
#### public getCancellationDetails() : \YooKassa\Model\Invoice\InvoiceCancellationDetails|null

```php
public getCancellationDetails() : \YooKassa\Model\Invoice\InvoiceCancellationDetails|null
```

**Summary**

Возвращает cancellation_details.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

**Returns:** \YooKassa\Model\Invoice\InvoiceCancellationDetails|null - Комментарий к статусу `canceled`


<a name="method_getCart" class="anchor"></a>
#### public getCart() : \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface|null

```php
public getCart() : \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface|null
```

**Summary**

Возвращает cart.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

**Returns:** \YooKassa\Model\Invoice\LineItem[]|\YooKassa\Common\ListObjectInterface|null - Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает created_at.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

**Returns:** \DateTime|null - Дата и время создания счета на оплату


<a name="method_getDeliveryMethod" class="anchor"></a>
#### public getDeliveryMethod() : \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|\YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodSelf|null

```php
public getDeliveryMethod() : \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|\YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodSelf|null
```

**Summary**

Возвращает delivery_method.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

**Returns:** \YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|\YooKassa\Model\Invoice\DeliveryMethod\DeliveryMethodSelf|null - Данные о выбранном способе доставки счета


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает description.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

**Returns:** string|null - Описание выставленного счета


<a name="method_getExpiresAt" class="anchor"></a>
#### public getExpiresAt() : \DateTime|null

```php
public getExpiresAt() : \DateTime|null
```

**Summary**

Возвращает expires_at.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

**Returns:** \DateTime|null - Срок действия счета


<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает id.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

**Returns:** string|null - Идентификатор счета в ЮKassa


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает metadata.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

**Returns:** \YooKassa\Model\Metadata|null - Любые дополнительные данные


<a name="method_getPaymentDetails" class="anchor"></a>
#### public getPaymentDetails() : \YooKassa\Model\Invoice\PaymentDetails|null

```php
public getPaymentDetails() : \YooKassa\Model\Invoice\PaymentDetails|null
```

**Summary**

Возвращает payment_details.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

**Returns:** \YooKassa\Model\Invoice\PaymentDetails|null - Данные о платеже по выставленному счету


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает status.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

**Returns:** string|null - Статус счета


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : bool

```php
public offsetExists(string $offset) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(string $offset) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : void

```php
public offsetSet(string $offset, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : void

```php
public offsetUnset(string $offset) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_setCancellationDetails" class="anchor"></a>
#### public setCancellationDetails() : self

```php
public setCancellationDetails(\YooKassa\Model\Invoice\InvoiceCancellationDetails|array|null $cancellation_details = null) : self
```

**Summary**

Устанавливает cancellation_details.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Invoice\InvoiceCancellationDetails OR array OR null</code> | cancellation_details  | Комментарий к статусу `canceled` |

**Returns:** self - 


<a name="method_setCart" class="anchor"></a>
#### public setCart() : self

```php
public setCart(\YooKassa\Common\ListObjectInterface|\YooKassa\Model\Invoice\LineItem[]|array|null $cart = null) : self
```

**Summary**

Устанавливает cart.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR \YooKassa\Model\Invoice\LineItem[] OR array OR null</code> | cart  | Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой |

**Returns:** self - 


<a name="method_setCreatedAt" class="anchor"></a>
#### public setCreatedAt() : self

```php
public setCreatedAt(\DateTime|string|null $created_at = null) : self
```

**Summary**

Устанавливает created_at.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | created_at  | Дата и время создания счета на оплату |

**Returns:** self - 


<a name="method_setDeliveryMethod" class="anchor"></a>
#### public setDeliveryMethod() : self

```php
public setDeliveryMethod(\YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod|array|null $delivery_method = null) : self
```

**Summary**

Устанавливает delivery_method.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Invoice\DeliveryMethod\AbstractDeliveryMethod OR array OR null</code> | delivery_method  | Данные о выбранном способе доставки счета |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает description.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37». |

**Returns:** self - 


<a name="method_setExpiresAt" class="anchor"></a>
#### public setExpiresAt() : self

```php
public setExpiresAt(\DateTime|array|null $expires_at = null) : self
```

**Summary**

Устанавливает expires_at.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR array OR null</code> | expires_at  | Срок действия счета |

**Returns:** self - 


<a name="method_setId" class="anchor"></a>
#### public setId() : self

```php
public setId(string|null $id = null) : self
```

**Summary**

Устанавливает id.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | id  | Идентификатор счета в ЮKassa. |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(\YooKassa\Model\Metadata|array|null $metadata = null) : self
```

**Summary**

Устанавливает metadata.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Metadata OR array OR null</code> | metadata  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). |

**Returns:** self - 


<a name="method_setPaymentDetails" class="anchor"></a>
#### public setPaymentDetails() : self

```php
public setPaymentDetails(\YooKassa\Model\Invoice\PaymentDetails|array|null $payment_details = null) : self
```

**Summary**

Устанавливает payment_details.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Invoice\PaymentDetails OR array OR null</code> | payment_details  | Данные о платеже по выставленному счету |

**Returns:** self - 


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : self

```php
public setStatus(string|null $status = null) : self
```

**Summary**

Устанавливает status.

**Details:**
* Inherited From: [\YooKassa\Model\Invoice\Invoice](../classes/YooKassa-Model-Invoice-Invoice.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | status  | Статус счета |

**Returns:** self - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
Является алиасом метода AbstractObject::jsonSerialize().

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_getUnknownProperties" class="anchor"></a>
#### protected getUnknownProperties() : array

```php
protected getUnknownProperties() : array
```

**Summary**

Возвращает массив свойств которые не существуют, но были заданы у объекта.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив с не существующими у текущего объекта свойствами


<a name="method_validatePropertyValue" class="anchor"></a>
#### protected validatePropertyValue() : mixed

```php
protected validatePropertyValue(string $propertyName, mixed $propertyValue) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  |  |
| <code lang="php">mixed</code> | propertyValue  |  |

**Returns:** mixed - 



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