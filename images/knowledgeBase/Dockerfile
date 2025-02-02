# Base image
FROM busybox

# Define working directory
WORKDIR /tmp/knowledgeBase/

# Copy AI model and Readme file
COPY currentAISolution.keras .
COPY currentOlsSolution.pkl .
COPY Readme.md .

# Set default command
CMD ["sh"]