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

namespace YooKassa\Request\PaymentMethods;

use YooKassa\Common\AbstractRequest;
use YooKassa\Model\SavePaymentMethod\SavePaymentMethodType;
use YooKassa\Request\PaymentMethods\ConfirmationData\AbstractConfirmation;
use YooKassa\Request\PaymentMethods\ConfirmationData\ConfirmationFactory;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель CreatePaymentMethodRequest.
 *
 * Данные для проверки и сохранения способа оплаты.
 * 
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта.
 * @property PaymentMethodCard $card Данные банковской карты (необходимы, если вы собираете данные карты пользователей на своей стороне).
 * @property PaymentMethodHolder $holder Данные магазина, для которого сохраняется способ оплаты.
 * @property string $client_ip IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.
 * @property string $clientIp IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.
 * @property AbstractConfirmation $confirmation Данные, необходимые для инициирования сценария подтверждения привязки.
*/
class CreatePaymentMethodRequest extends AbstractRequest implements CreatePaymentMethodRequestInterface
{
    /**
     * Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [SavePaymentMethodType::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_type = null;

    /**
     * Данные для проверки и сохранения банковской карты.
     *
     * @var PaymentMethodCard|null
     */
    #[Assert\Valid]
    #[Assert\Type(PaymentMethodCard::class)]
    private ?PaymentMethodCard $_card = null;

    /**
     * Данные магазина, для которого сохраняется способ оплаты.
     *
     * @var PaymentMethodHolder|null
     */
    #[Assert\Type(PaymentMethodHolder::class)]
    private ?PaymentMethodHolder $_holder = null;

    /**
     * IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_client_ip = null;

    /**
     * Данные, необходимые для инициирования сценария подтверждения привязки.
     *
     * @var AbstractConfirmation|null
     */
    #[Assert\Type(AbstractConfirmation::class)]
    private ?AbstractConfirmation $_confirmation = null;

    /**
     * Возвращает type.
     *
     * @return string|null
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает type.
     *
     * @param string|null $type Код способа оплаты. Возможное значение: ~`bank_card` — банковская карта.
     *
     * @return self
     */
    public function setType(?string $type = null): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        return $this;
    }

    /**
     * Проверяет, были ли установлен код способа оплаты.
     *
     * @return bool True если код способа оплаты был установлен, false если нет
     */
    public function hasType(): bool
    {
        return null !== $this->_type;
    }

    /**
     * Возвращает card.
     *
     * @return PaymentMethodCard|null
     */
    public function getCard(): ?PaymentMethodCard
    {
        return $this->_card;
    }

    /**
     * Устанавливает card.
     *
     * @param PaymentMethodCard|array|null $card Данные для проверки и сохранения банковской карты.
     *
     * @return self
     */
    public function setCard(mixed $card): self
    {
        $this->_card = $this->validatePropertyValue('_card', $card);
        return $this;
    }

    /**
     * Возвращает holder.
     *
     * @return PaymentMethodHolder|null
     */
    public function getHolder(): ?PaymentMethodHolder
    {
        return $this->_holder;
    }

    /**
     * Устанавливает holder.
     *
     * @param PaymentMethodHolder|array|null $holder Данные магазина, для которого сохраняется способ оплаты.
     *
     * @return self
     */
    public function setHolder(mixed $holder = null): self
    {
        $this->_holder = $this->validatePropertyValue('_holder', $holder);
        return $this;
    }

    /**
     * Возвращает client_ip.
     *
     * @return string|null
     */
    public function getClientIp(): ?string
    {
        return $this->_client_ip;
    }

    /**
     * Устанавливает client_ip.
     *
     * @param string|null $client_ip IPv4 или IPv6-адрес пользователя. Если не указан, используется IP-адрес TCP-подключения.
     *
     * @return self
     */
    public function setClientIp(?string $client_ip = null): self
    {
        $this->_client_ip = $this->validatePropertyValue('_client_ip', $client_ip);
        return $this;
    }

    /**
     * Возвращает confirmation.
     *
     * @return AbstractConfirmation|null
     */
    public function getConfirmation(): ?AbstractConfirmation
    {
        return $this->_confirmation;
    }

    /**
     * Устанавливает confirmation.
     *
     * @param AbstractConfirmation|array|null $confirmation Данные, необходимые для инициирования сценария подтверждения привязки.
     *
     * @return self
     */
    public function setConfirmation(mixed $confirmation = null): self
    {
        if (is_array($confirmation)) {
            $confirmation = (new ConfirmationFactory())->factoryFromArray($confirmation);
        }
        $this->_confirmation = $this->validatePropertyValue('_confirmation', $confirmation);
        return $this;
    }

    /**
     * Валидирует текущий запрос, проверяет все ли нужные свойства установлены.
     *
     * @return bool True если запрос валиден, false если нет
     */
    public function validate(): bool
    {
        if (!$this->hasType()) {
            $this->setValidationError('Type not specified');
            return false;
        }

        return true;
    }

    /**
     * Возвращает билдер объектов запросов создания платежа.
     *
     * @return CreatePaymentMethodRequestBuilder Инстанс билдера объектов запросов
     */
    public static function builder(): CreatePaymentMethodRequestBuilder
    {
        return new CreatePaymentMethodRequestBuilder();
    }
}

