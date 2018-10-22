import os
import random
import discord
from lib.text_generation import create_ode, NUMBER_OF_ITERATIONS

client = discord.Client()


@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')


@client.event
async def on_message(message):
    if message.content.startswith('!ode'):
        await client.send_message(message.channel, 'Идет творческий процесс...')
        ode = create_ode()
        await client.send_message(message.channel, f'Ода Пульсу №{random.randint(1, 10000)}:')
        await client.send_message(message.channel, ode)
    elif message.content.startswith('!hi'):
        await client.send_message(
            message.channel,
            f'Hello {message.author.mention}. I am Pulse Bot! Use `!commands` to check all of my commands'
        )
    elif message.content.startswith('!commands'):
        await client.send_message(
            message.channel,
            f'List of the commands: ```!hi - intro command\n!ode - genereate ode to Pulse\n!info - information about Bot\n!home - Where do I live```'
        )
    elif message.content.startswith('!info'):
        await client.send_message(
            message.channel,
            f'These bot generate odes to Pulse, using neural network trained on Pushkin\'s poems (currently with {NUMBER_OF_ITERATIONS} iterations)'
        )
    elif message.content.startswith('!home'):
        await client.send_message(
            message.channel,
            f'Code for bot you can find here: https://github.com/Gogen120/PulseBot'
        )


if __name__ == '__main__':
    client.run(os.environ['DISCORD_TOKEN'])
