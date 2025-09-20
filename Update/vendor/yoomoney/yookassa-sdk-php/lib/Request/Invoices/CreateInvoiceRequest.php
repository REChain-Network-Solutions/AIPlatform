<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Request\Invoices;

use DateTime;
use YooKassa\Common\AbstractRequest;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Invoice\LineItem;
use YooKassa\Model\Metadata;
use YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData;
use YooKassa\Request\Invoices\DeliveryMethodData\DeliveryMethodDataFactory;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель CreateInvoiceRequest.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property PaymentData $payment_data Данные для проведения платежа по выставленному счету.
 * @property LineItem[]|ListObjectInterface $cart Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.
 * @property AbstractDeliveryMethodData $delivery_method_data Данные о способе доставки счета пользователю. Доступен только один способ — самостоятельная доставка: ЮKassa возвращает вам ссылку на счет, и вы передаете ее пользователю любым удобным для вас способом.
 * @property DateTime $expires_at Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z`
 * @property string $locale Язык интерфейса, писем и смс, которые будет видеть или получать пользователь. Формат соответствует [ISO/IEC 15897](https://en.wikipedia.org/wiki/Locale_(computer_software)). Возможные значения: ru_RU, en_US. Регистр важен.
 * @property string $description Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37».
 * @property Metadata $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
*/
class CreateInvoiceRequest extends AbstractRequest implements CreateInvoiceRequestInterface
{
    /**
     * Данные для проведения платежа по выставленному счету.
     *
     * @var PaymentData|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\Type(PaymentData::class)]
    private ?PaymentData $_payment_data = null;

    /**
     * Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.
     *
     * @var LineItem[]|ListObjectInterface|null
     */
    #[Assert\NotBlank]
    #[Assert\Valid]
    #[Assert\AllType(LineItem::class)]
    #[Assert\Type(ListObject::class)]
    private ?ListObjectInterface $_cart = null;

    /**
     * Данные о способе доставки счета пользователю.
     * Доступен только один способ — самостоятельная доставка: ЮKassa возвращает вам ссылку на счет, и вы передаете ее пользователю любым удобным для вас способом.
     *
     * @var AbstractDeliveryMethodData|null
     */
    #[Assert\Type(AbstractDeliveryMethodData::class)]
    private ?AbstractDeliveryMethodData $_delivery_method_data = null;

    /**
     * Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z`
     *
     * @var DateTime|null
     */
    #[Assert\NotBlank]
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_expires_at = null;

    /**
     * Язык интерфейса, писем и смс, которые будет видеть или получать пользователь.
     * Формат соответствует [ISO/IEC 15897](https://en.wikipedia.org/wiki/Locale_(computer_software)).
     * Возможные значения: ru_RU, en_US. Регистр важен.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_locale = null;

    /**
     * Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета.
     * Например: «Счет на оплату по договору 37».
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 128)]
    private ?string $_description = null;

    /**
     * Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа).
     * Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa.
     * Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @var Metadata|null
     */
    #[Assert\Type(Metadata::class)]
    private ?Metadata $_metadata = null;

    /**
     * Возвращает payment_data.
     *
     * @return PaymentData|null Данные для проведения платежа по выставленному счету
     */
    public function getPaymentData(): ?PaymentData
    {
        return $this->_payment_data;
    }

    /**
     * Проверяет, были ли установлены данные для проведения платежа.
     *
     * @return bool True если данные для проведения платежа были установлены, false если нет
     */
    public function hasPaymentData(): bool
    {
        return null !== $this->_payment_data;
    }

    /**
     * Устанавливает payment_data.
     *
     * @param PaymentData|array|null $payment_data Данные для проведения платежа по выставленному счету
     *
     * @return self
     */
    public function setPaymentData(mixed $payment_data = null): self
    {
        $this->_payment_data = $this->validatePropertyValue('_payment_data', $payment_data);
        return $this;
    }

    /**
     * Возвращает cart.
     *
     * @return LineItem[]|ListObjectInterface Корзина заказа — список товаров или услуг
     */
    public function getCart(): ListObjectInterface
    {
        if ($this->_cart === null) {
            $this->_cart = new ListObject(LineItem::class);
        }
        return $this->_cart;
    }

    /**
     * Проверяет, были ли установлены товары в корзине.
     *
     * @return bool True если товары в корзине были установлены, false если нет
     */
    public function hasCart(): bool
    {
        return $this->getCart()->count() > 0;
    }

    /**
     * Устанавливает cart.
     *
     * @param ListObjectInterface|array|null $cart Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой.
     *
     * @return self
     */
    public function setCart(mixed $cart = null): self
    {
        $this->_cart = $this->validatePropertyValue('_cart', $cart);
        return $this;
    }

    /**
     * Возвращает delivery_method_data.
     *
     * @return AbstractDeliveryMethodData|null Данные о способе доставки счета пользователю
     */
    public function getDeliveryMethodData(): ?AbstractDeliveryMethodData
    {
        return $this->_delivery_method_data;
    }

    /**
     * Проверяет, были ли установлены данные для о способе доставки счета.
     *
     * @return bool True если данные о способе доставки счета были установлены, false если нет
     */
    public function hasDeliveryMethodData(): bool
    {
        return null !== $this->_delivery_method_data;
    }

    /**
     * Устанавливает delivery_method_data.
     *
     * @param AbstractDeliveryMethodData|array|null $delivery_method_data Данные о способе доставки счета пользователю
     *
     * @return self
     */
    public function setDeliveryMethodData(mixed $delivery_method_data = null): self
    {
        if (is_array($delivery_method_data)) {
            $delivery_method_data = (new DeliveryMethodDataFactory())->factoryFromArray($delivery_method_data);
        }
        $this->_delivery_method_data = $this->validatePropertyValue('_delivery_method_data', $delivery_method_data);
        return $this;
    }

    /**
     * Возвращает expires_at.
     *
     * @return DateTime|null Срок действия счета
     */
    public function getExpiresAt(): ?DateTime
    {
        return $this->_expires_at;
    }

    /**
     * Проверяет, был ли установлен срок действия счета.
     *
     * @return bool True если срок действия счета был установлен, false если нет
     */
    public function hasExpiresAt(): bool
    {
        return null !== $this->_expires_at;
    }

    /**
     * Устанавливает expires_at.
     *
     * @param DateTime|string|null $expires_at Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z`
     *
     * @return self
     */
    public function setExpiresAt(DateTime|string|null $expires_at = null): self
    {
        $this->_expires_at = $this->validatePropertyValue('_expires_at', $expires_at);
        return $this;
    }

    /**
     * Возвращает locale.
     *
     * @return string|null Язык интерфейса, писем и смс, которые будет видеть или получать пользователь
     */
    public function getLocale(): ?string
    {
        return $this->_locale;
    }

    /**
     * Проверяет наличие языка интерфейса в создаваемом счете.
     *
     * @return bool True если язык интерфейса есть, false если нет
     */
    public function hasLocale(): bool
    {
        return !empty($this->_locale);
    }

    /**
     * Устанавливает locale.
     *
     * @param string|null $locale Язык интерфейса, писем и смс, которые будет видеть или получать пользователь
     *
     * @return self
     */
    public function setLocale(?string $locale = null): self
    {
        $this->_locale = $this->validatePropertyValue('_locale', $locale);
        return $this;
    }

    /**
     * Возвращает description.
     *
     * @return string|null Описание выставленного счета
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Проверяет наличие описания транзакции в создаваемом счете.
     *
     * @return bool True если описание транзакции есть, false если нет
     */
    public function hasDescription(): bool
    {
        return !empty($this->_description);
    }

    /**
     * Устанавливает description.
     *
     * @param string|null $description Описание выставленного счета (не более 128 символов), которое вы увидите в личном кабинете ЮKassa, а пользователь на странице счета. Например: «Счет на оплату по договору 37».
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Возвращает metadata.
     *
     * @return Metadata|null Любые дополнительные данные
     */
    public function getMetadata(): ?Metadata
    {
        return $this->_metadata;
    }

    /**
     * Проверяет, были ли установлены метаданные счета.
     *
     * @return bool True если метаданные были установлены, false если нет
     */
    public function hasMetadata(): bool
    {
        return !empty($this->_metadata) && $this->_metadata->count() > 0;
    }

    /**
     * Устанавливает metadata.
     *
     * @param Metadata|array|null $metadata Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.
     *
     * @return self
     */
    public function setMetadata(mixed $metadata = null): self
    {
        $this->_metadata = $this->validatePropertyValue('_metadata', $metadata);
        return $this;
    }

    /**
     * Проверяет на валидность текущий объект
     *
     * @return bool True если объект запроса валиден, false если нет
     */
    public function validate(): bool
    {
        if (!$this->hasPaymentData()) {
            $this->setValidationError('PaymentData not specified');
            return false;
        }
        if (null === $this->getPaymentData()?->getAmount()) {
            $this->setValidationError('PaymentData amount not specified');
            return false;
        }
        $value = $this->getPaymentData()?->getAmount()?->getValue();
        if (empty($value) || $value <= 0.0) {
            $this->setValidationError('Invalid PaymentData amount value: ' . $value);
            return false;
        }
        if ($this->getPaymentData()?->getReceipt()?->notEmpty()) {
            $email = $this->getPaymentData()?->getReceipt()?->getCustomer()?->getEmail();
            $phone = $this->getPaymentData()?->getReceipt()?->getCustomer()?->getPhone();
            if (empty($email) && empty($phone)) {
                $this->setValidationError('Both email and phone values are empty in receipt');

                return false;
            }
        }
        if (!$this->hasCart()) {
            $this->setValidationError('Cart not specified');
            return false;
        }
        if (!$this->hasExpiresAt()) {
            $this->setValidationError('ExpiresAt not specified');
            return false;
        }

        return true;
    }

    /**
     * Возвращает билдер объектов запросов создания платежа.
     *
     * @return CreateInvoiceRequestBuilder Инстанс билдера объектов запросов
     */
    public static function builder(): CreateInvoiceRequestBuilder
    {
        return new CreateInvoiceRequestBuilder();
    }
}
