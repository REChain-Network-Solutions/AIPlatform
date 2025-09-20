# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payout\Payout
### Namespace: [\YooKassa\Model\Payout](../namespaces/yookassa-model-payout.md)
---
**Summary:**

Класс, представляющий модель Payout.

**Description:**

Объект выплаты (Payout) — актуальная информация о выплате.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MAX_LENGTH_DESCRIPTION](../classes/YooKassa-Model-Payout-Payout.md#constant_MAX_LENGTH_DESCRIPTION) |  |  |
| public | [MAX_LENGTH_ID](../classes/YooKassa-Model-Payout-Payout.md#constant_MAX_LENGTH_ID) |  |  |
| public | [MIN_LENGTH_ID](../classes/YooKassa-Model-Payout-Payout.md#constant_MIN_LENGTH_ID) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$amount](../classes/YooKassa-Model-Payout-Payout.md#property_amount) |  | Сумма выплаты |
| public | [$cancellation_details](../classes/YooKassa-Model-Payout-Payout.md#property_cancellation_details) |  | Комментарий к отмене выплаты |
| public | [$cancellationDetails](../classes/YooKassa-Model-Payout-Payout.md#property_cancellationDetails) |  | Комментарий к отмене выплаты |
| public | [$created_at](../classes/YooKassa-Model-Payout-Payout.md#property_created_at) |  | Время создания заказа |
| public | [$createdAt](../classes/YooKassa-Model-Payout-Payout.md#property_createdAt) |  | Время создания заказа |
| public | [$deal](../classes/YooKassa-Model-Payout-Payout.md#property_deal) |  | Сделка, в рамках которой нужно провести выплату |
| public | [$description](../classes/YooKassa-Model-Payout-Payout.md#property_description) |  | Описание транзакции |
| public | [$id](../classes/YooKassa-Model-Payout-Payout.md#property_id) |  | Идентификатор выплаты |
| public | [$metadata](../classes/YooKassa-Model-Payout-Payout.md#property_metadata) |  | Метаданные выплаты указанные мерчантом |
| public | [$payout_destination](../classes/YooKassa-Model-Payout-Payout.md#property_payout_destination) |  | Способ проведения выплаты |
| public | [$payoutDestination](../classes/YooKassa-Model-Payout-Payout.md#property_payoutDestination) |  | Способ проведения выплаты |
| public | [$receipt](../classes/YooKassa-Model-Payout-Payout.md#property_receipt) |  | Данные чека, зарегистрированного в ФНС |
| public | [$self_employed](../classes/YooKassa-Model-Payout-Payout.md#property_self_employed) |  | Данные самозанятого, который получит выплату |
| public | [$selfEmployed](../classes/YooKassa-Model-Payout-Payout.md#property_selfEmployed) |  | Данные самозанятого, который получит выплату |
| public | [$status](../classes/YooKassa-Model-Payout-Payout.md#property_status) |  | Текущее состояние выплаты |
| public | [$succeeded_at](../classes/YooKassa-Model-Payout-Payout.md#property_succeeded_at) |  | Время успешного проведения выплаты |
| public | [$succeededAt](../classes/YooKassa-Model-Payout-Payout.md#property_succeededAt) |  | Время успешного проведения выплаты |
| public | [$test](../classes/YooKassa-Model-Payout-Payout.md#property_test) |  | Признак тестовой операции |
| protected | [$_amount](../classes/YooKassa-Model-Payout-Payout.md#property__amount) |  | Сумма выплаты |
| protected | [$_cancellation_details](../classes/YooKassa-Model-Payout-Payout.md#property__cancellation_details) |  | Комментарий к статусу canceled: кто отменил выплаты и по какой причине |
| protected | [$_created_at](../classes/YooKassa-Model-Payout-Payout.md#property__created_at) |  | Время создания выплаты. Пример: ~`2017-11-03T11:52:31.827Z` |
| protected | [$_deal](../classes/YooKassa-Model-Payout-Payout.md#property__deal) |  | Сделка, в рамках которой нужно провести выплату. Присутствует, если вы проводите Безопасную сделку |
| protected | [$_description](../classes/YooKassa-Model-Payout-Payout.md#property__description) |  | Описание транзакции (не более 128 символов). Например: «Выплата по договору 37». |
| protected | [$_id](../classes/YooKassa-Model-Payout-Payout.md#property__id) |  | Идентификатор выплаты. |
| protected | [$_metadata](../classes/YooKassa-Model-Payout-Payout.md#property__metadata) |  | Метаданные выплаты указанные мерчантом |
| protected | [$_payout_destination](../classes/YooKassa-Model-Payout-Payout.md#property__payout_destination) |  | Способ проведения выплаты |
| protected | [$_receipt](../classes/YooKassa-Model-Payout-Payout.md#property__receipt) |  | Данные чека, зарегистрированного в ФНС. Присутствует, если вы делаете выплату самозанятому. |
| protected | [$_self_employed](../classes/YooKassa-Model-Payout-Payout.md#property__self_employed) |  | Данные самозанятого, который получит выплату. Присутствует, если вы делаете выплату самозанятому |
| protected | [$_status](../classes/YooKassa-Model-Payout-Payout.md#property__status) |  | Текущее состояние выплаты |
| protected | [$_succeeded_at](../classes/YooKassa-Model-Payout-Payout.md#property__succeeded_at) |  | Время успешного проведения выплаты. Пример: ~`2017-11-03T11:52:42.312Z` Обязательный параметр для выплат в статусе ~`succeeded` |
| protected | [$_test](../classes/YooKassa-Model-Payout-Payout.md#property__test) |  | Признак тестовой операции |

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
| public | [getAmount()](../classes/YooKassa-Model-Payout-Payout.md#method_getAmount) |  | Возвращает сумму. |
| public | [getCancellationDetails()](../classes/YooKassa-Model-Payout-Payout.md#method_getCancellationDetails) |  | Возвращает комментарий к статусу canceled: кто отменил платеж и по какой причине. |
| public | [getCreatedAt()](../classes/YooKassa-Model-Payout-Payout.md#method_getCreatedAt) |  | Возвращает время создания выплаты. |
| public | [getDeal()](../classes/YooKassa-Model-Payout-Payout.md#method_getDeal) |  | Возвращает сделку, в рамках которой нужно провести выплату. |
| public | [getDescription()](../classes/YooKassa-Model-Payout-Payout.md#method_getDescription) |  | Возвращает описание транзакции. |
| public | [getId()](../classes/YooKassa-Model-Payout-Payout.md#method_getId) |  | Возвращает идентификатор выплаты. |
| public | [getMetadata()](../classes/YooKassa-Model-Payout-Payout.md#method_getMetadata) |  | Возвращает метаданные выплаты установленные мерчантом |
| public | [getPayoutDestination()](../classes/YooKassa-Model-Payout-Payout.md#method_getPayoutDestination) |  | Возвращает используемый способ проведения выплаты. |
| public | [getReceipt()](../classes/YooKassa-Model-Payout-Payout.md#method_getReceipt) |  | Возвращает данные чека, зарегистрированного в ФНС. |
| public | [getSelfEmployed()](../classes/YooKassa-Model-Payout-Payout.md#method_getSelfEmployed) |  | Возвращает данные самозанятого, который получит выплату. |
| public | [getStatus()](../classes/YooKassa-Model-Payout-Payout.md#method_getStatus) |  | Возвращает состояние выплаты. |
| public | [getSucceededAt()](../classes/YooKassa-Model-Payout-Payout.md#method_getSucceededAt) |  | Возвращает время успешного проведения выплаты. |
| public | [getTest()](../classes/YooKassa-Model-Payout-Payout.md#method_getTest) |  | Возвращает признак тестовой операции. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAmount()](../classes/YooKassa-Model-Payout-Payout.md#method_setAmount) |  | Устанавливает сумму выплаты. |
| public | [setCancellationDetails()](../classes/YooKassa-Model-Payout-Payout.md#method_setCancellationDetails) |  | Устанавливает комментарий к статусу canceled: кто отменил платеж и по какой причине. |
| public | [setCreatedAt()](../classes/YooKassa-Model-Payout-Payout.md#method_setCreatedAt) |  | Устанавливает время создания выплаты. |
| public | [setDeal()](../classes/YooKassa-Model-Payout-Payout.md#method_setDeal) |  | Устанавливает сделку, в рамках которой нужно провести выплату. |
| public | [setDescription()](../classes/YooKassa-Model-Payout-Payout.md#method_setDescription) |  | Устанавливает описание транзакции |
| public | [setId()](../classes/YooKassa-Model-Payout-Payout.md#method_setId) |  | Устанавливает идентификатор выплаты. |
| public | [setMetadata()](../classes/YooKassa-Model-Payout-Payout.md#method_setMetadata) |  | Устанавливает метаданные выплаты. |
| public | [setPayoutDestination()](../classes/YooKassa-Model-Payout-Payout.md#method_setPayoutDestination) |  | Устанавливает используемый способ проведения выплаты. |
| public | [setReceipt()](../classes/YooKassa-Model-Payout-Payout.md#method_setReceipt) |  | Устанавливает данные чека, зарегистрированного в ФНС. |
| public | [setSelfEmployed()](../classes/YooKassa-Model-Payout-Payout.md#method_setSelfEmployed) |  | Устанавливает данные самозанятого, который получит выплату. |
| public | [setStatus()](../classes/YooKassa-Model-Payout-Payout.md#method_setStatus) |  | Устанавливает статус выплаты |
| public | [setSucceededAt()](../classes/YooKassa-Model-Payout-Payout.md#method_setSucceededAt) |  | Устанавливает время успешного проведения выплаты. |
| public | [setTest()](../classes/YooKassa-Model-Payout-Payout.md#method_setTest) |  | Устанавливает признак тестовой операции. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Payout/Payout.php](../../lib/Model/Payout/Payout.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Payout\Payout
* Implements:
  * [\YooKassa\Model\Payout\PayoutInterface](../classes/YooKassa-Model-Payout-PayoutInterface.md)

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
<a name="constant_MAX_LENGTH_DESCRIPTION" class="anchor"></a>
###### MAX_LENGTH_DESCRIPTION
```php
MAX_LENGTH_DESCRIPTION = 128 : int
```


<a name="constant_MAX_LENGTH_ID" class="anchor"></a>
###### MAX_LENGTH_ID
```php
MAX_LENGTH_ID = 50 : int
```


<a name="constant_MIN_LENGTH_ID" class="anchor"></a>
###### MIN_LENGTH_ID
```php
MIN_LENGTH_ID = 36 : int
```



---
## Properties
<a name="property_amount"></a>
#### public $amount : \YooKassa\Model\AmountInterface
---
***Description***

Сумма выплаты

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_cancellation_details"></a>
#### public $cancellation_details : \YooKassa\Model\CancellationDetailsInterface
---
***Description***

Комментарий к отмене выплаты

**Type:** <a href="../classes/YooKassa-Model-CancellationDetailsInterface.html"><abbr title="\YooKassa\Model\CancellationDetailsInterface">CancellationDetailsInterface</abbr></a>

**Details:**


<a name="property_cancellationDetails"></a>
#### public $cancellationDetails : \YooKassa\Model\CancellationDetailsInterface
---
***Description***

Комментарий к отмене выплаты

**Type:** <a href="../classes/YooKassa-Model-CancellationDetailsInterface.html"><abbr title="\YooKassa\Model\CancellationDetailsInterface">CancellationDetailsInterface</abbr></a>

**Details:**


<a name="property_created_at"></a>
#### public $created_at : \DateTime
---
***Description***

Время создания заказа

**Type:** \DateTime

**Details:**


<a name="property_createdAt"></a>
#### public $createdAt : \DateTime
---
***Description***

Время создания заказа

**Type:** \DateTime

**Details:**


<a name="property_deal"></a>
#### public $deal : \YooKassa\Model\Deal\PayoutDealInfo
---
***Description***

Сделка, в рамках которой нужно провести выплату

**Type:** <a href="../classes/YooKassa-Model-Deal-PayoutDealInfo.html"><abbr title="\YooKassa\Model\Deal\PayoutDealInfo">PayoutDealInfo</abbr></a>

**Details:**


<a name="property_description"></a>
#### public $description : string
---
***Description***

Описание транзакции

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_id"></a>
#### public $id : string
---
***Description***

Идентификатор выплаты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : \YooKassa\Model\Metadata
---
***Description***

Метаданные выплаты указанные мерчантом

**Type:** <a href="../classes/YooKassa-Model-Metadata.html"><abbr title="\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property_payout_destination"></a>
#### public $payout_destination : \YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod
---
***Description***

Способ проведения выплаты

**Type:** <a href="../classes/YooKassa-Model-Payment-PaymentMethod-AbstractPaymentMethod.html"><abbr title="\YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod">AbstractPaymentMethod</abbr></a>

**Details:**


<a name="property_payoutDestination"></a>
#### public $payoutDestination : \YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod
---
***Description***

Способ проведения выплаты

**Type:** <a href="../classes/YooKassa-Model-Payment-PaymentMethod-AbstractPaymentMethod.html"><abbr title="\YooKassa\Model\Payment\PaymentMethod\AbstractPaymentMethod">AbstractPaymentMethod</abbr></a>

**Details:**


<a name="property_receipt"></a>
#### public $receipt : \YooKassa\Model\Payout\IncomeReceipt
---
***Description***

Данные чека, зарегистрированного в ФНС

**Type:** <a href="../classes/YooKassa-Model-Payout-IncomeReceipt.html"><abbr title="\YooKassa\Model\Payout\IncomeReceipt">IncomeReceipt</abbr></a>

**Details:**


<a name="property_self_employed"></a>
#### public $self_employed : \YooKassa\Model\Payout\PayoutSelfEmployed
---
***Description***

Данные самозанятого, который получит выплату

**Type:** <a href="../classes/YooKassa-Model-Payout-PayoutSelfEmployed.html"><abbr title="\YooKassa\Model\Payout\PayoutSelfEmployed">PayoutSelfEmployed</abbr></a>

**Details:**


<a name="property_selfEmployed"></a>
#### public $selfEmployed : \YooKassa\Model\Payout\PayoutSelfEmployed
---
***Description***

Данные самозанятого, который получит выплату

**Type:** <a href="../classes/YooKassa-Model-Payout-PayoutSelfEmployed.html"><abbr title="\YooKassa\Model\Payout\PayoutSelfEmployed">PayoutSelfEmployed</abbr></a>

**Details:**


<a name="property_status"></a>
#### public $status : string
---
***Description***

Текущее состояние выплаты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_succeeded_at"></a>
#### public $succeeded_at : \DateTime
---
***Description***

Время успешного проведения выплаты

**Type:** \DateTime

**Details:**


<a name="property_succeededAt"></a>
#### public $succeededAt : \DateTime
---
***Description***

Время успешного проведения выплаты

**Type:** \DateTime

**Details:**


<a name="property_test"></a>
#### public $test : bool
---
***Description***

Признак тестовой операции

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property__amount"></a>
#### protected $_amount : ?\YooKassa\Model\AmountInterface
---
**Summary**

Сумма выплаты

**Type:** <a href="../?\YooKassa\Model\AmountInterface"><abbr title="?\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property__cancellation_details"></a>
#### protected $_cancellation_details : ?\YooKassa\Model\Payout\PayoutCancellationDetails
---
**Summary**

Комментарий к статусу canceled: кто отменил выплаты и по какой причине

**Type:** <a href="../?\YooKassa\Model\Payout\PayoutCancellationDetails"><abbr title="?\YooKassa\Model\Payout\PayoutCancellationDetails">PayoutCancellationDetails</abbr></a>

**Details:**


<a name="property__created_at"></a>
#### protected $_created_at : ?\DateTime
---
**Summary**

Время создания выплаты. Пример: ~`2017-11-03T11:52:31.827Z`

**Type:** <a href="../?\DateTime"><abbr title="?\DateTime">DateTime</abbr></a>

**Details:**


<a name="property__deal"></a>
#### protected $_deal : ?\YooKassa\Model\Deal\PayoutDealInfo
---
**Summary**

Сделка, в рамках которой нужно провести выплату. Присутствует, если вы проводите Безопасную сделку

**Type:** <a href="../?\YooKassa\Model\Deal\PayoutDealInfo"><abbr title="?\YooKassa\Model\Deal\PayoutDealInfo">PayoutDealInfo</abbr></a>

**Details:**


<a name="property__description"></a>
#### protected $_description : ?string
---
**Summary**

Описание транзакции (не более 128 символов). Например: «Выплата по договору 37».

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**


<a name="property__id"></a>
#### protected $_id : ?string
---
**Summary**

Идентификатор выплаты.

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**


<a name="property__metadata"></a>
#### protected $_metadata : ?\YooKassa\Model\Metadata
---
**Summary**

Метаданные выплаты указанные мерчантом

**Type:** <a href="../?\YooKassa\Model\Metadata"><abbr title="?\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property__payout_destination"></a>
#### protected $_payout_destination : ?\YooKassa\Model\Payout\AbstractPayoutDestination
---
**Summary**

Способ проведения выплаты

**Type:** <a href="../?\YooKassa\Model\Payout\AbstractPayoutDestination"><abbr title="?\YooKassa\Model\Payout\AbstractPayoutDestination">AbstractPayoutDestination</abbr></a>

**Details:**


<a name="property__receipt"></a>
#### protected $_receipt : ?\YooKassa\Model\Payout\IncomeReceipt
---
**Summary**

Данные чека, зарегистрированного в ФНС. Присутствует, если вы делаете выплату самозанятому.

**Type:** <a href="../?\YooKassa\Model\Payout\IncomeReceipt"><abbr title="?\YooKassa\Model\Payout\IncomeReceipt">IncomeReceipt</abbr></a>

**Details:**


<a name="property__self_employed"></a>
#### protected $_self_employed : ?\YooKassa\Model\Payout\PayoutSelfEmployed
---
**Summary**

Данные самозанятого, который получит выплату. Присутствует, если вы делаете выплату самозанятому

**Type:** <a href="../?\YooKassa\Model\Payout\PayoutSelfEmployed"><abbr title="?\YooKassa\Model\Payout\PayoutSelfEmployed">PayoutSelfEmployed</abbr></a>

**Details:**


<a name="property__status"></a>
#### protected $_status : ?string
---
**Summary**

Текущее состояние выплаты

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**


<a name="property__succeeded_at"></a>
#### protected $_succeeded_at : ?\DateTime
---
**Summary**

Время успешного проведения выплаты. Пример: ~`2017-11-03T11:52:42.312Z`
Обязательный параметр для выплат в статусе ~`succeeded`

**Type:** <a href="../?\DateTime"><abbr title="?\DateTime">DateTime</abbr></a>

**Details:**


<a name="property__test"></a>
#### protected $_test : ?bool
---
**Summary**

Признак тестовой операции

**Type:** <a href="../?bool"><abbr title="?bool">?bool</abbr></a>

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


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма выплаты


<a name="method_getCancellationDetails" class="anchor"></a>
#### public getCancellationDetails() : \YooKassa\Model\Payout\PayoutCancellationDetails|null

```php
public getCancellationDetails() : \YooKassa\Model\Payout\PayoutCancellationDetails|null
```

**Summary**

Возвращает комментарий к статусу canceled: кто отменил платеж и по какой причине.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** \YooKassa\Model\Payout\PayoutCancellationDetails|null - Комментарий к статусу canceled


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает время создания выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** \DateTime|null - Время создания выплаты


<a name="method_getDeal" class="anchor"></a>
#### public getDeal() : \YooKassa\Model\Deal\PayoutDealInfo|null

```php
public getDeal() : \YooKassa\Model\Deal\PayoutDealInfo|null
```

**Summary**

Возвращает сделку, в рамках которой нужно провести выплату.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** \YooKassa\Model\Deal\PayoutDealInfo|null - Сделка, в рамках которой нужно провести выплату


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** string|null - 


<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает идентификатор выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** string|null - Идентификатор выплаты


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает метаданные выплаты установленные мерчантом

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** \YooKassa\Model\Metadata|null - Метаданные выплаты указанные мерчантом


<a name="method_getPayoutDestination" class="anchor"></a>
#### public getPayoutDestination() : \YooKassa\Model\Payout\AbstractPayoutDestination|null

```php
public getPayoutDestination() : \YooKassa\Model\Payout\AbstractPayoutDestination|null
```

**Summary**

Возвращает используемый способ проведения выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** \YooKassa\Model\Payout\AbstractPayoutDestination|null - Способ проведения выплаты


<a name="method_getReceipt" class="anchor"></a>
#### public getReceipt() : \YooKassa\Model\Payout\IncomeReceipt|null

```php
public getReceipt() : \YooKassa\Model\Payout\IncomeReceipt|null
```

**Summary**

Возвращает данные чека, зарегистрированного в ФНС.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** \YooKassa\Model\Payout\IncomeReceipt|null - Данные чека, зарегистрированного в ФНС


<a name="method_getSelfEmployed" class="anchor"></a>
#### public getSelfEmployed() : \YooKassa\Model\Payout\PayoutSelfEmployed|null

```php
public getSelfEmployed() : \YooKassa\Model\Payout\PayoutSelfEmployed|null
```

**Summary**

Возвращает данные самозанятого, который получит выплату.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** \YooKassa\Model\Payout\PayoutSelfEmployed|null - Данные самозанятого, который получит выплату


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает состояние выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** string|null - Текущее состояние выплаты


<a name="method_getSucceededAt" class="anchor"></a>
#### public getSucceededAt() : \DateTime|null

```php
public getSucceededAt() : \DateTime|null
```

**Summary**

Возвращает время успешного проведения выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** \DateTime|null - Время успешного проведения выплаты


<a name="method_getTest" class="anchor"></a>
#### public getTest() : bool|null

```php
public getTest() : bool|null
```

**Summary**

Возвращает признак тестовой операции.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

**Returns:** bool|null - Признак тестовой операции


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


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array|null $amount = null) : self
```

**Summary**

Устанавливает сумму выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | amount  | Сумма выплаты |

**Returns:** self - 


<a name="method_setCancellationDetails" class="anchor"></a>
#### public setCancellationDetails() : self

```php
public setCancellationDetails(\YooKassa\Model\Payout\PayoutCancellationDetails|array|null $cancellation_details = null) : self
```

**Summary**

Устанавливает комментарий к статусу canceled: кто отменил платеж и по какой причине.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Payout\PayoutCancellationDetails OR array OR null</code> | cancellation_details  | Комментарий к статусу canceled |

**Returns:** self - 


<a name="method_setCreatedAt" class="anchor"></a>
#### public setCreatedAt() : self

```php
public setCreatedAt(\DateTime|string|null $created_at = null) : self
```

**Summary**

Устанавливает время создания выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | created_at  | Время создания выплаты. Пример: ~`2017-11-03T11:52:31.827Z` |

**Returns:** self - 


<a name="method_setDeal" class="anchor"></a>
#### public setDeal() : self

```php
public setDeal(\YooKassa\Model\Deal\PayoutDealInfo|array|null $deal = null) : self
```

**Summary**

Устанавливает сделку, в рамках которой нужно провести выплату.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Deal\PayoutDealInfo OR array OR null</code> | deal  | Сделка, в рамках которой нужно провести выплату |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает описание транзакции

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание транзакции (не более 128 символов). Например: «Выплата по договору 37». |

**Returns:** self - 


<a name="method_setId" class="anchor"></a>
#### public setId() : self

```php
public setId(string|null $id = null) : self
```

**Summary**

Устанавливает идентификатор выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | id  | Идентификатор выплаты |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(\YooKassa\Model\Metadata|array|null $metadata = null) : self
```

**Summary**

Устанавливает метаданные выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Metadata OR array OR null</code> | metadata  | Метаданные выплаты указанные мерчантом |

**Returns:** self - 


<a name="method_setPayoutDestination" class="anchor"></a>
#### public setPayoutDestination() : $this

```php
public setPayoutDestination(\YooKassa\Model\Payout\AbstractPayoutDestination|array|null $payout_destination) : $this
```

**Summary**

Устанавливает используемый способ проведения выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Payout\AbstractPayoutDestination OR array OR null</code> | payout_destination  | Способ проведения выплаты |

**Returns:** $this - 


<a name="method_setReceipt" class="anchor"></a>
#### public setReceipt() : self

```php
public setReceipt(\YooKassa\Model\Payout\IncomeReceipt|array|null $receipt = null) : self
```

**Summary**

Устанавливает данные чека, зарегистрированного в ФНС.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Payout\IncomeReceipt OR array OR null</code> | receipt  | Данные чека, зарегистрированного в ФНС |

**Returns:** self - 


<a name="method_setSelfEmployed" class="anchor"></a>
#### public setSelfEmployed() : self

```php
public setSelfEmployed(\YooKassa\Model\Payout\PayoutSelfEmployed|array|null $self_employed = null) : self
```

**Summary**

Устанавливает данные самозанятого, который получит выплату.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Payout\PayoutSelfEmployed OR array OR null</code> | self_employed  | Данные самозанятого, который получит выплату |

**Returns:** self - 


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : self

```php
public setStatus(string|null $status = null) : self
```

**Summary**

Устанавливает статус выплаты

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | status  | Статус выплаты |

**Returns:** self - 


<a name="method_setSucceededAt" class="anchor"></a>
#### public setSucceededAt() : self

```php
public setSucceededAt(\DateTime|string|null $succeeded_at = null) : self
```

**Summary**

Устанавливает время успешного проведения выплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | succeeded_at  | Время успешного проведения выплаты. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2017-11-03T11:52:42.312Z`  Обязательный параметр для выплат в статусе ~`succeeded`. |

**Returns:** self - 


<a name="method_setTest" class="anchor"></a>
#### public setTest() : self

```php
public setTest(bool|null $test = null) : self
```

**Summary**

Устанавливает признак тестовой операции.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\Payout](../classes/YooKassa-Model-Payout-Payout.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool OR null</code> | test  | Признак тестовой операции |

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