# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Deal\SafeDeal
### Namespace: [\YooKassa\Model\Deal](../namespaces/yookassa-model-deal.md)
---
**Summary:**

Класс, представляющий модель SafeDeal.


---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MAX_LENGTH_ID](../classes/YooKassa-Model-Deal-SafeDeal.md#constant_MAX_LENGTH_ID) |  |  |
| public | [MIN_LENGTH_ID](../classes/YooKassa-Model-Deal-SafeDeal.md#constant_MIN_LENGTH_ID) |  |  |
| public | [MAX_LENGTH_DESCRIPTION](../classes/YooKassa-Model-Deal-SafeDeal.md#constant_MAX_LENGTH_DESCRIPTION) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$balance](../classes/YooKassa-Model-Deal-SafeDeal.md#property_balance) |  | Баланс сделки. |
| public | [$created_at](../classes/YooKassa-Model-Deal-SafeDeal.md#property_created_at) |  | Время создания сделки. |
| public | [$createdAt](../classes/YooKassa-Model-Deal-SafeDeal.md#property_createdAt) |  | Время создания сделки. |
| public | [$description](../classes/YooKassa-Model-Deal-SafeDeal.md#property_description) |  | Описание сделки (не более 128 символов). |
| public | [$expires_at](../classes/YooKassa-Model-Deal-SafeDeal.md#property_expires_at) |  | Время автоматического закрытия сделки. |
| public | [$expiresAt](../classes/YooKassa-Model-Deal-SafeDeal.md#property_expiresAt) |  | Время автоматического закрытия сделки. |
| public | [$fee_moment](../classes/YooKassa-Model-Deal-SafeDeal.md#property_fee_moment) |  | Момент перечисления вам вознаграждения платформы. |
| public | [$feeMoment](../classes/YooKassa-Model-Deal-SafeDeal.md#property_feeMoment) |  | Момент перечисления вам вознаграждения платформы. |
| public | [$id](../classes/YooKassa-Model-Deal-SafeDeal.md#property_id) |  | Идентификатор сделки. |
| public | [$metadata](../classes/YooKassa-Model-Deal-SafeDeal.md#property_metadata) |  | Любые дополнительные данные, которые нужны вам для работы. |
| public | [$payout_balance](../classes/YooKassa-Model-Deal-SafeDeal.md#property_payout_balance) |  | Сумма вознаграждения продавцаю. |
| public | [$payoutBalance](../classes/YooKassa-Model-Deal-SafeDeal.md#property_payoutBalance) |  | Сумма вознаграждения продавца. |
| public | [$status](../classes/YooKassa-Model-Deal-SafeDeal.md#property_status) |  | Статус сделки. |
| public | [$test](../classes/YooKassa-Model-Deal-SafeDeal.md#property_test) |  | Признак тестовой операции. |
| public | [$type](../classes/YooKassa-Model-Deal-AbstractBaseDeal.md#property_type) |  | Тип сделки |
| protected | [$_balance](../classes/YooKassa-Model-Deal-SafeDeal.md#property__balance) |  | Баланс сделки |
| protected | [$_created_at](../classes/YooKassa-Model-Deal-SafeDeal.md#property__created_at) |  | Время создания сделки. |
| protected | [$_description](../classes/YooKassa-Model-Deal-SafeDeal.md#property__description) |  | Описание сделки (не более 128 символов). Используется для фильтрации при [получении списка сделок](/developers/api#get_deals_list). |
| protected | [$_expires_at](../classes/YooKassa-Model-Deal-SafeDeal.md#property__expires_at) |  | Время автоматического закрытия сделки. |
| protected | [$_fee_moment](../classes/YooKassa-Model-Deal-SafeDeal.md#property__fee_moment) |  | Момент перечисления вам вознаграждения платформы |
| protected | [$_id](../classes/YooKassa-Model-Deal-SafeDeal.md#property__id) |  | Идентификатор сделки. |
| protected | [$_metadata](../classes/YooKassa-Model-Deal-SafeDeal.md#property__metadata) |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). |
| protected | [$_payout_balance](../classes/YooKassa-Model-Deal-SafeDeal.md#property__payout_balance) |  | Сумма вознаграждения продавца |
| protected | [$_status](../classes/YooKassa-Model-Deal-SafeDeal.md#property__status) |  | Статус сделки |
| protected | [$_test](../classes/YooKassa-Model-Deal-SafeDeal.md#property__test) |  | Признак тестовой операции. |
| protected | [$_type](../classes/YooKassa-Model-Deal-AbstractBaseDeal.md#property__type) |  | Тип сделки |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Model-Deal-SafeDeal.md#method___construct) |  | Конструктор SafeDeal. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getBalance()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_getBalance) |  | Возвращает баланс сделки. |
| public | [getCreatedAt()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_getCreatedAt) |  | Возвращает время создания сделки. |
| public | [getDescription()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_getDescription) |  | Возвращает описание сделки (не более 128 символов). |
| public | [getExpiresAt()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_getExpiresAt) |  | Возвращает время автоматического закрытия сделки. |
| public | [getFeeMoment()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_getFeeMoment) |  | Возвращает момент перечисления вам вознаграждения платформы. |
| public | [getId()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_getId) |  | Возвращает Id сделки. |
| public | [getMetadata()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_getMetadata) |  | Возвращает дополнительные данные сделки. |
| public | [getPayoutBalance()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_getPayoutBalance) |  | Возвращает сумму вознаграждения продавца. |
| public | [getStatus()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_getStatus) |  | Возвращает статус сделки. |
| public | [getTest()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_getTest) |  | Возвращает признак тестовой операции. |
| public | [getType()](../classes/YooKassa-Model-Deal-AbstractBaseDeal.md#method_getType) |  | Возвращает тип сделки. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setBalance()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_setBalance) |  | Устанавливает баланс сделки. |
| public | [setCreatedAt()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_setCreatedAt) |  | Устанавливает created_at. |
| public | [setDescription()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_setDescription) |  | Устанавливает описание сделки (не более 128 символов). |
| public | [setExpiresAt()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_setExpiresAt) |  | Устанавливает время автоматического закрытия сделки. |
| public | [setFeeMoment()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_setFeeMoment) |  | Устанавливает момент перечисления вам вознаграждения платформы. |
| public | [setId()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_setId) |  | Устанавливает Id сделки. |
| public | [setMetadata()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_setMetadata) |  | Устанавливает metadata. |
| public | [setPayoutBalance()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_setPayoutBalance) |  | Устанавливает сумму вознаграждения продавца. |
| public | [setStatus()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_setStatus) |  | Устанавливает статус сделки. |
| public | [setTest()](../classes/YooKassa-Model-Deal-SafeDeal.md#method_setTest) |  | Устанавливает признак тестовой операции. |
| public | [setType()](../classes/YooKassa-Model-Deal-AbstractBaseDeal.md#method_setType) |  | Устанавливает тип сделки. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Deal/SafeDeal.php](../../lib/Model/Deal/SafeDeal.php)
* Package: YooKassa\Model
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Model\Deal\AbstractBaseDeal](../classes/YooKassa-Model-Deal-AbstractBaseDeal.md)
  * \YooKassa\Model\Deal\SafeDeal
* Implements:
  * [\YooKassa\Model\Deal\DealInterface](../classes/YooKassa-Model-Deal-DealInterface.md)

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


<a name="constant_MAX_LENGTH_DESCRIPTION" class="anchor"></a>
###### MAX_LENGTH_DESCRIPTION
```php
MAX_LENGTH_DESCRIPTION = 128 : int
```



---
## Properties
<a name="property_balance"></a>
#### public $balance : \YooKassa\Model\AmountInterface
---
***Description***

Баланс сделки.

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_created_at"></a>
#### public $created_at : \DateTime
---
***Description***

Время создания сделки.

**Type:** \DateTime

**Details:**


<a name="property_createdAt"></a>
#### public $createdAt : \DateTime
---
***Description***

Время создания сделки.

**Type:** \DateTime

**Details:**


<a name="property_description"></a>
#### public $description : string
---
***Description***

Описание сделки (не более 128 символов).

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_expires_at"></a>
#### public $expires_at : \DateTime
---
***Description***

Время автоматического закрытия сделки.

**Type:** \DateTime

**Details:**


<a name="property_expiresAt"></a>
#### public $expiresAt : \DateTime
---
***Description***

Время автоматического закрытия сделки.

**Type:** \DateTime

**Details:**


<a name="property_fee_moment"></a>
#### public $fee_moment : string
---
***Description***

Момент перечисления вам вознаграждения платформы.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_feeMoment"></a>
#### public $feeMoment : string
---
***Description***

Момент перечисления вам вознаграждения платформы.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_id"></a>
#### public $id : string
---
***Description***

Идентификатор сделки.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : \YooKassa\Model\Metadata
---
***Description***

Любые дополнительные данные, которые нужны вам для работы.

**Type:** <a href="../classes/YooKassa-Model-Metadata.html"><abbr title="\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property_payout_balance"></a>
#### public $payout_balance : \YooKassa\Model\AmountInterface
---
***Description***

Сумма вознаграждения продавцаю.

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_payoutBalance"></a>
#### public $payoutBalance : \YooKassa\Model\AmountInterface
---
***Description***

Сумма вознаграждения продавца.

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_status"></a>
#### public $status : string
---
***Description***

Статус сделки.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_test"></a>
#### public $test : bool
---
***Description***

Признак тестовой операции.

**Type:** <a href="../bool"><abbr title="bool">bool</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип сделки

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Deal\AbstractBaseDeal](../classes/YooKassa-Model-Deal-AbstractBaseDeal.md)


<a name="property__balance"></a>
#### protected $_balance : ?\YooKassa\Model\AmountInterface
---
**Summary**

Баланс сделки

**Type:** <a href="../?\YooKassa\Model\AmountInterface"><abbr title="?\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property__created_at"></a>
#### protected $_created_at : ?\DateTime
---
**Summary**

Время создания сделки.

**Type:** <a href="../?\DateTime"><abbr title="?\DateTime">DateTime</abbr></a>

**Details:**


<a name="property__description"></a>
#### protected $_description : ?string
---
**Summary**

Описание сделки (не более 128 символов). Используется для фильтрации при [получении списка сделок](/developers/api#get_deals_list).

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**


<a name="property__expires_at"></a>
#### protected $_expires_at : ?\DateTime
---
**Summary**

Время автоматического закрытия сделки.

**Type:** <a href="../?\DateTime"><abbr title="?\DateTime">DateTime</abbr></a>

**Details:**


<a name="property__fee_moment"></a>
#### protected $_fee_moment : ?string
---
**Summary**

Момент перечисления вам вознаграждения платформы

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**


<a name="property__id"></a>
#### protected $_id : ?string
---
**Summary**

Идентификатор сделки.

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**


<a name="property__metadata"></a>
#### protected $_metadata : ?\YooKassa\Model\Metadata
---
**Summary**

Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).

**Type:** <a href="../?\YooKassa\Model\Metadata"><abbr title="?\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property__payout_balance"></a>
#### protected $_payout_balance : ?\YooKassa\Model\AmountInterface
---
**Summary**

Сумма вознаграждения продавца

**Type:** <a href="../?\YooKassa\Model\AmountInterface"><abbr title="?\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property__status"></a>
#### protected $_status : ?string
---
**Summary**

Статус сделки

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**


<a name="property__test"></a>
#### protected $_test : ?bool
---
**Summary**

Признак тестовой операции.

**Type:** <a href="../?bool"><abbr title="?bool">?bool</abbr></a>

**Details:**


<a name="property__type"></a>
#### protected $_type : ?string
---
**Summary**

Тип сделки

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Deal\AbstractBaseDeal](../classes/YooKassa-Model-Deal-AbstractBaseDeal.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

Конструктор SafeDeal.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

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


<a name="method_getBalance" class="anchor"></a>
#### public getBalance() : \YooKassa\Model\AmountInterface|null

```php
public getBalance() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает баланс сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Баланс сделки


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает время создания сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

**Returns:** \DateTime|null - Время создания сделки


<a name="method_getDescription" class="anchor"></a>
#### public getDescription() : string|null

```php
public getDescription() : string|null
```

**Summary**

Возвращает описание сделки (не более 128 символов).

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

**Returns:** string|null - Описание сделки


<a name="method_getExpiresAt" class="anchor"></a>
#### public getExpiresAt() : \DateTime|null

```php
public getExpiresAt() : \DateTime|null
```

**Summary**

Возвращает время автоматического закрытия сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

**Returns:** \DateTime|null - Время автоматического закрытия сделки


<a name="method_getFeeMoment" class="anchor"></a>
#### public getFeeMoment() : string|null

```php
public getFeeMoment() : string|null
```

**Summary**

Возвращает момент перечисления вам вознаграждения платформы.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

**Returns:** string|null - Момент перечисления вознаграждения


<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает Id сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

**Returns:** string|null - Id сделки


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает дополнительные данные сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

**Returns:** \YooKassa\Model\Metadata|null - Дополнительные данные сделки


<a name="method_getPayoutBalance" class="anchor"></a>
#### public getPayoutBalance() : \YooKassa\Model\AmountInterface|null

```php
public getPayoutBalance() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму вознаграждения продавца.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма вознаграждения продавца


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает статус сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

**Returns:** string|null - Статус сделки


<a name="method_getTest" class="anchor"></a>
#### public getTest() : bool|null

```php
public getTest() : bool|null
```

**Summary**

Возвращает признак тестовой операции.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

**Returns:** bool|null - Признак тестовой операции


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\AbstractBaseDeal](../classes/YooKassa-Model-Deal-AbstractBaseDeal.md)

**Returns:** string|null - 


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


<a name="method_setBalance" class="anchor"></a>
#### public setBalance() : self

```php
public setBalance(\YooKassa\Model\AmountInterface|array|null $balance = null) : self
```

**Summary**

Устанавливает баланс сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | balance  |  |

**Returns:** self - 


<a name="method_setCreatedAt" class="anchor"></a>
#### public setCreatedAt() : self

```php
public setCreatedAt(\DateTime|string|null $created_at = null) : self
```

**Summary**

Устанавливает created_at.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | created_at  | Время создания сделки. |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : self

```php
public setDescription(string|null $description = null) : self
```

**Summary**

Устанавливает описание сделки (не более 128 символов).

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | description  | Описание сделки (не более 128 символов). |

**Returns:** self - 


<a name="method_setExpiresAt" class="anchor"></a>
#### public setExpiresAt() : self

```php
public setExpiresAt(\DateTime|string|null $expires_at = null) : self
```

**Summary**

Устанавливает время автоматического закрытия сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | expires_at  | Время автоматического закрытия сделки. |

**Returns:** self - 


<a name="method_setFeeMoment" class="anchor"></a>
#### public setFeeMoment() : self

```php
public setFeeMoment(string|null $fee_moment = null) : self
```

**Summary**

Устанавливает момент перечисления вам вознаграждения платформы.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | fee_moment  | Момент перечисления вам вознаграждения платформы |

**Returns:** self - 


<a name="method_setId" class="anchor"></a>
#### public setId() : self

```php
public setId(string|null $id = null) : self
```

**Summary**

Устанавливает Id сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | id  | Идентификатор сделки. |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(\YooKassa\Model\Metadata|array|null $metadata = null) : self
```

**Summary**

Устанавливает metadata.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Metadata OR array OR null</code> | metadata  | Любые дополнительные данные, которые нужны вам для работы. |

**Returns:** self - 


<a name="method_setPayoutBalance" class="anchor"></a>
#### public setPayoutBalance() : self

```php
public setPayoutBalance(\YooKassa\Model\AmountInterface|array|null $payout_balance = null) : self
```

**Summary**

Устанавливает сумму вознаграждения продавца.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | payout_balance  | Сумма вознаграждения продавца |

**Returns:** self - 


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : self

```php
public setStatus(string|null $status = null) : self
```

**Summary**

Устанавливает статус сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | status  | Статус сделки |

**Returns:** self - 


<a name="method_setTest" class="anchor"></a>
#### public setTest() : self

```php
public setTest(bool|null $test = null) : self
```

**Summary**

Устанавливает признак тестовой операции.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\SafeDeal](../classes/YooKassa-Model-Deal-SafeDeal.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool OR null</code> | test  | Признак тестовой операции |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $type = null) : self
```

**Summary**

Устанавливает тип сделки.

**Details:**
* Inherited From: [\YooKassa\Model\Deal\AbstractBaseDeal](../classes/YooKassa-Model-Deal-AbstractBaseDeal.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  |  |

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